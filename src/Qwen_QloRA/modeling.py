import torch

from Qwen_QloRA.constants import ID_TO_LABEL, LABEL_TO_ID


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


def load_qwen_sequence_classifier(
    model_path: str,
    load_in_4bit: bool,
    dtype: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    gradient_checkpointing: bool,
):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    quantization_config = None
    if load_in_4bit:
        compute_dtype = torch.float16 if dtype == "auto" else torch_dtype(dtype)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL_TO_ID),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        trust_remote_code=True,
        torch_dtype=torch_dtype(dtype),
        quantization_config=quantization_config,
        device_map="auto" if load_in_4bit else None,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

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

