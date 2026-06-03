import inspect
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from hardware_profiles import kbit_device_map
from Gemma2_QLoRA.constants import ID_TO_LABEL, LABEL_TO_ID


def torch_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    if dtype == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype}")


def maybe_disable_softcapping(config, disable_softcapping: bool):
    if disable_softcapping:
        if hasattr(config, "attn_logit_softcapping"):
            config.attn_logit_softcapping = None
        if hasattr(config, "final_logit_softcapping"):
            config.final_logit_softcapping = None
    return config


def parse_target_modules(value: str) -> str | list[str]:
    if value == "all-linear":
        return value
    return [module.strip() for module in value.split(",") if module.strip()]


class MLPClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float,
        hidden_ratio: float,
    ) -> None:
        super().__init__()
        head_hidden_size = max(num_labels, int(hidden_size * hidden_ratio))
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, num_labels),
        )

    def forward(self, hidden_states):
        return self.net(hidden_states)


def replace_classification_head(
    model,
    head_type: str,
    dropout: float,
    hidden_ratio: float,
):
    if head_type == "linear":
        return model
    if head_type != "mlp":
        raise ValueError(f"Unsupported classifier head: {head_type}")

    old_score_parameter = next(model.score.parameters(), None)
    device = old_score_parameter.device if old_score_parameter is not None else None
    dtype = old_score_parameter.dtype if old_score_parameter is not None else None

    new_score = MLPClassificationHead(
        hidden_size=model.config.hidden_size,
        num_labels=model.config.num_labels,
        dropout=dropout,
        hidden_ratio=hidden_ratio,
    )
    if device is not None and dtype is not None:
        new_score = new_score.to(device=device, dtype=dtype)
    model.score = new_score
    return model


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


@contextmanager
def peft_adapter_with_supported_config(adapter_path: str, config_cls):
    adapter_dir = Path(adapter_path)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        yield adapter_path
        return

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    supported_keys = set(inspect.signature(config_cls.__init__).parameters)
    supported_keys.discard("self")
    filtered_config = {
        key: value
        for key, value in adapter_config.items()
        if key in supported_keys or key == "peft_type"
    }
    removed_keys = sorted(set(adapter_config) - set(filtered_config))
    if not removed_keys:
        yield adapter_path
        return

    with tempfile.TemporaryDirectory(prefix="peft_adapter_") as tmp:
        tmp_dir = Path(tmp)
        for item in adapter_dir.iterdir():
            target = tmp_dir / item.name
            if item.name == "adapter_config.json":
                continue
            os.symlink(item.resolve(), target, target_is_directory=item.is_dir())
        target_config_path = tmp_dir / "adapter_config.json"
        target_config_path.write_text(
            json.dumps(filtered_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            "  resume_adapter_config: ignored unsupported PEFT keys: "
            + ", ".join(removed_keys)
        )
        yield str(tmp_dir)


def load_gemma2_sequence_classifier(
    model_path: str,
    resume_adapter: str | None,
    load_in_4bit: bool,
    dtype: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str | list[str],
    gradient_checkpointing: bool,
    disable_softcapping: bool,
    classifier_head: str,
    head_dropout: float,
    head_hidden_ratio: float,
):
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_labels = len(LABEL_TO_ID)
    config.id2label = ID_TO_LABEL
    config.label2id = LABEL_TO_ID
    config.use_cache = False
    config = maybe_disable_softcapping(config, disable_softcapping)

    quantization_config = None
    device_map = None
    if load_in_4bit:
        compute_dtype = torch.float16 if dtype == "auto" else torch_dtype(dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        device_map = kbit_device_map()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype(dtype),
        quantization_config=quantization_config,
        device_map=device_map,
        ignore_mismatched_sizes=True,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.config = maybe_disable_softcapping(model.config, disable_softcapping)
    model = replace_classification_head(
        model,
        head_type=classifier_head,
        dropout=head_dropout,
        hidden_ratio=head_hidden_ratio,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    if resume_adapter is not None:
        with peft_adapter_with_supported_config(resume_adapter, LoraConfig) as adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=target_modules,
            modules_to_save=["score"],
        )
        model = get_peft_model(model, lora_config)
    return model
