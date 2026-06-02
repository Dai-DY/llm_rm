import argparse
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
from Qwen_QloRA.constants import DEFAULT_MODEL_PATH, LABEL_COLUMNS
from Qwen_QloRA.data import DataCollatorForPreference, PreferenceDataset
from Qwen_QloRA.metrics import softmax
from Qwen_QloRA.modeling import load_tokenizer, torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict A/B/tie probabilities with a trained Qwen LoRA adapter."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Base Qwen model path.")
    parser.add_argument("--adapter", required=True, help="Trained LoRA adapter directory.")
    parser.add_argument("--input", required=True, help="Input CSV.")
    parser.add_argument("--output", required=True, help="Output probability CSV.")
    parser.add_argument("--max-length", type=int, default=1800, help="Max token length.")
    add_hardware_profile_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument(
        "--has-labels",
        action="store_true",
        help="Set for validation CSVs that contain labels.",
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
    return apply_profile_defaults(parser.parse_args(), "qwen_predict")


def load_model(args: argparse.Namespace):
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

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
        num_labels=len(LABEL_COLUMNS),
        trust_remote_code=True,
        torch_dtype=torch_dtype(args.dtype),
        quantization_config=quantization_config,
        device_map=device_map,
    )
    if base.config.pad_token_id is None:
        base.config.pad_token_id = base.config.eos_token_id
    model = PeftModel.from_pretrained(base, args.adapter)
    if not args.load_in_4bit and torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    print(f"hardware_profile: {args.hardware_profile} ({args.hardware_description})")
    print(f"batch_size: {args.batch_size}")
    print(f"dtype: {args.dtype}")
    print(f"load_in_4bit: {args.load_in_4bit}")
    tokenizer = load_tokenizer(args.adapter)
    dataset = PreferenceDataset(
        csv_path=args.input,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.limit,
        has_labels=args.has_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForPreference(tokenizer),
    )
    model = load_model(args)

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            row_ids = batch.pop("id")
            batch.pop("labels", None)
            device = model_input_device(model)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.detach().float().cpu().numpy()
            probabilities = softmax(logits)
            for row_id, probs in zip(row_ids, probabilities):
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
