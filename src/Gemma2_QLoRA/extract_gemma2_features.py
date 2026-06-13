import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hardware_profiles import (
    add_hardware_profile_argument,
    apply_profile_defaults,
    kbit_device_map,
    model_input_device,
)
from Gemma2_QLoRA.constants import DEFAULT_MODEL_PATH, LABEL_COLUMNS
from Gemma2_QLoRA.data import DataCollatorForPreference, PreferenceDataset
from Gemma2_QLoRA.modeling import (
    maybe_disable_softcapping,
    peft_adapter_with_supported_config,
    replace_classification_head,
    torch_dtype,
)
from Gemma2_QLoRA.modeling import load_tokenizer
from Gemma2_QLoRA.train import pooled_last_hidden


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pooled Gemma2+LoRA hidden features once, so the MLP "
            "classification head can be trained separately without repeatedly "
            "running the full model."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Base Gemma2 RM path.")
    parser.add_argument("--adapter", required=True, help="Trained LoRA adapter directory.")
    parser.add_argument("--input", required=True, help="Input CSV.")
    parser.add_argument("--output", required=True, help="Output .pt feature file.")
    parser.add_argument("--max-length", type=int, default=1800, help="Max token length.")
    add_hardware_profile_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--has-labels", action="store_true", help="Input CSV has labels.")
    parser.add_argument(
        "--swap-augmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Duplicate labeled rows with response A/B swapped before extracting train features.",
    )
    parser.add_argument(
        "--swap-inputs",
        action="store_true",
        help="Extract features from A/B-swapped inputs without changing labels.",
    )
    parser.add_argument(
        "--disable-softcapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set Gemma2 softcapping config fields to None.",
    )
    parser.add_argument(
        "--classifier-head",
        choices=["linear", "mlp"],
        default=None,
        help="Classification head architecture used when loading the adapter.",
    )
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument("--head-hidden-ratio", type=float, default=None)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=None,
        help="Model compute dtype.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use bitsandbytes 4-bit loading.",
    )
    return apply_profile_defaults(parser.parse_args(), "gemma_predict")


def apply_adapter_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    config_path = Path(args.adapter) / "gemma2_qlora_config.json"
    saved_config = {}
    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.classifier_head is None:
        args.classifier_head = saved_config.get("classifier_head", "mlp")
    if args.head_dropout is None:
        args.head_dropout = float(saved_config.get("head_dropout", 0.1))
    if args.head_hidden_ratio is None:
        args.head_hidden_ratio = float(saved_config.get("head_hidden_ratio", 0.5))
    return args


def load_model(args: argparse.Namespace):
    from peft import LoraConfig, PeftModel
    from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.num_labels = len(LABEL_COLUMNS)
    config.output_hidden_states = True
    config = maybe_disable_softcapping(config, args.disable_softcapping)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.float16 if args.dtype == "auto" else torch_dtype(args.dtype)
            ),
            bnb_4bit_use_double_quant=True,
        )
    device_map = kbit_device_map() if args.load_in_4bit else None

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        quantization_config=quantization_config,
        device_map=device_map,
        ignore_mismatched_sizes=True,
    )
    if base.config.pad_token_id is None:
        base.config.pad_token_id = base.config.eos_token_id
    base.config = maybe_disable_softcapping(base.config, args.disable_softcapping)
    base = replace_classification_head(
        base,
        head_type=args.classifier_head,
        dropout=args.head_dropout,
        hidden_ratio=args.head_hidden_ratio,
    )
    with peft_adapter_with_supported_config(args.adapter, LoraConfig) as adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
    if not args.load_in_4bit and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def extract_features(model, dataset: PreferenceDataset, tokenizer, batch_size: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForPreference(tokenizer),
    )
    ids = []
    feature_batches = []
    label_batches = []
    device = model_input_device(model)
    with torch.no_grad():
        for batch in tqdm(loader, desc="extract"):
            row_ids = batch.pop("id")
            labels = batch.pop("labels", None)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            features = pooled_last_hidden(outputs, batch["attention_mask"])
            ids.extend(row_ids)
            feature_batches.append(features.detach().float().cpu())
            if labels is not None:
                label_batches.append(labels.detach().long().cpu())

    labels = torch.cat(label_batches, dim=0) if label_batches else None
    return ids, torch.cat(feature_batches, dim=0), labels


def main() -> None:
    args = apply_adapter_config_defaults(parse_args())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"hardware_profile: {args.hardware_profile} ({args.hardware_description})")
    print(f"model: {args.model}")
    print(f"adapter: {args.adapter}")
    print(f"input: {args.input}")
    print(f"output: {output_path}")
    print(f"batch_size: {args.batch_size}")
    print(f"dtype: {args.dtype}")
    print(f"load_in_4bit: {args.load_in_4bit}")
    print(f"max_length: {args.max_length}")
    print(f"swap_augmentation: {args.swap_augmentation}")
    print(f"swap_inputs: {args.swap_inputs}")

    tokenizer = load_tokenizer(args.adapter)
    model = load_model(args)
    dataset = PreferenceDataset(
        csv_path=args.input,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.limit,
        has_labels=args.has_labels,
        swap_augmentation=args.swap_augmentation,
        swap_inputs=args.swap_inputs,
    )
    ids, features, labels = extract_features(model, dataset, tokenizer, args.batch_size)
    print(f"rows: {len(ids)}")
    print(f"features: {tuple(features.shape)}")

    payload = {
        "ids": ids,
        "features": features,
        "labels": labels,
        "metadata": {
            "model": args.model,
            "adapter": args.adapter,
            "input": args.input,
            "max_length": args.max_length,
            "classifier_head": args.classifier_head,
            "head_dropout": args.head_dropout,
            "head_hidden_ratio": args.head_hidden_ratio,
            "disable_softcapping": args.disable_softcapping,
            "swap_augmentation": args.swap_augmentation,
            "swap_inputs": args.swap_inputs,
        },
    }
    torch.save(payload, output_path)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
