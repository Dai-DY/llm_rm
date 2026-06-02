import argparse
import json
from pathlib import Path

import pandas as pd
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
from Gemma2_QLoRA.metrics import softmax
from Gemma2_QLoRA.modeling import maybe_disable_softcapping, replace_classification_head, torch_dtype
from Gemma2_QLoRA.modeling import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict A/B/tie probabilities with a trained Gemma2 LoRA adapter."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Base Gemma2 RM path.")
    parser.add_argument("--adapter", required=True, help="Trained LoRA adapter directory.")
    parser.add_argument("--input", required=True, help="Input CSV.")
    parser.add_argument("--output", required=True, help="Output probability CSV.")
    parser.add_argument("--max-length", type=int, default=1800, help="Max token length.")
    add_hardware_profile_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--has-labels", action="store_true", help="Input has labels.")
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average original and A/B-flipped predictions.",
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
        help="Classification head architecture. Defaults to adapter config or mlp.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=None,
        help="Dropout used by the MLP classification head.",
    )
    parser.add_argument(
        "--head-hidden-ratio",
        type=float,
        default=None,
        help="MLP hidden size as a fraction of Gemma2 hidden size.",
    )
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
        help="Use bitsandbytes 4-bit loading. Disabled by default for 4090 bf16 LoRA.",
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
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForSequenceClassification, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.num_labels = len(LABEL_COLUMNS)
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
    model = PeftModel.from_pretrained(base, args.adapter)
    if not args.load_in_4bit and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model


def predict_dataset(model, dataset: PreferenceDataset, tokenizer, batch_size: int) -> tuple[list[str], list]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorForPreference(tokenizer),
    )
    ids = []
    probabilities = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            row_ids = batch.pop("id")
            batch.pop("labels", None)
            device = model_input_device(model)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.detach().float().cpu().numpy()
            ids.extend(row_ids)
            probabilities.extend(softmax(logits))
    return ids, probabilities


def main() -> None:
    args = parse_args()
    args = apply_adapter_config_defaults(args)
    print(f"hardware_profile: {args.hardware_profile} ({args.hardware_description})")
    print(f"batch_size: {args.batch_size}")
    print(f"dtype: {args.dtype}")
    print(f"load_in_4bit: {args.load_in_4bit}")
    tokenizer = load_tokenizer(args.adapter)
    model = load_model(args)

    dataset = PreferenceDataset(
        csv_path=args.input,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.limit,
        has_labels=args.has_labels,
    )
    ids, probabilities = predict_dataset(model, dataset, tokenizer, args.batch_size)

    if args.tta:
        swapped_dataset = PreferenceDataset(
            csv_path=args.input,
            tokenizer=tokenizer,
            max_length=args.max_length,
            limit=args.limit,
            has_labels=args.has_labels,
            swap_inputs=True,
        )
        swapped_ids, swapped_probabilities = predict_dataset(
            model,
            swapped_dataset,
            tokenizer,
            args.batch_size,
        )
        if ids != swapped_ids:
            raise ValueError("Original and swapped prediction ids do not match.")
        probabilities = [
            (original + swapped[[1, 0, 2]]) / 2.0
            for original, swapped in zip(probabilities, swapped_probabilities)
        ]

    rows = []
    for row_id, probs in zip(ids, probabilities):
        rows.append(
            {
                "id": row_id,
                LABEL_COLUMNS[0]: probs[0],
                LABEL_COLUMNS[1]: probs[1],
                LABEL_COLUMNS[2]: probs[2],
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
