import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from RM_LogisticRegression.rm_scoring import load_reward_model, score_dataframe
from RM_LogisticRegression.paths import default_score_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score response_a/response_b with a local Gemma2 reward model."
    )
    parser.add_argument(
        "--model",
        default="models/sfairXC__FsfairX-Gemma2-RM-v0.1",
        help="Local reward model path.",
    )
    parser.add_argument("--input", required=True, help="Input CSV to score.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output score CSV. Defaults to "
            "output/<run>/RM_LogisticRegression/<input_stem>_gemma_rm_scores.csv."
        ),
    )
    parser.add_argument(
        "--output-date",
        "--run-name",
        dest="output_date",
        default=None,
        help=(
            "Run folder for default output paths. Defaults to current time "
            "formatted as YYYY-MM-DD_HH-MM."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Token truncation length. Lower this if 8GB VRAM is tight.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load model in 4bit with bitsandbytes. Disabled by default for 4090 bf16 inference.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16"],
        default="bfloat16",
        help="Model compute dtype.",
    )
    parser.add_argument(
        "--gpu-memory",
        default="23GiB",
        help="Max memory for GPU 0 when using device_map=auto.",
    )
    parser.add_argument(
        "--cpu-memory",
        default="48GiB",
        help="Max CPU memory for device_map=auto offload.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Write partial output every N input rows.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip ids already present in the output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = (
        Path(args.output)
        if args.output is not None
        else default_score_output_path(input_path, args.limit, args.output_date)
    )

    print("[stage 1/5] Resolve paths and configuration")
    print(f"  input: {input_path}")
    print(f"  output: {output_path}")
    print(f"  model: {args.model}")
    print(f"  limit: {args.limit if args.limit is not None else 'none'}")
    print(f"  max_length: {args.max_length}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  load_in_4bit: {args.load_in_4bit}")
    print(f"  dtype: {args.dtype}")
    print(f"  gpu_memory: {args.gpu_memory}")
    print(f"  resume: {args.resume}")

    print("[stage 2/5] Read input CSV")
    df = pd.read_csv(input_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    print(f"  rows loaded: {len(df)}")

    required = ["id", "prompt", "response_a", "response_b"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing columns: {missing}")

    print("[stage 3/5] Load existing checkpoint rows")
    existing_rows = []
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        if "id" not in existing.columns:
            raise ValueError(f"{output_path} exists but does not contain an id column.")
        existing_rows = existing.to_dict("records")
        print(f"  resume: loaded {len(existing_rows)} existing scored rows from {output_path}")
    else:
        print("  no existing checkpoint loaded")

    print("[stage 4/5] Load reward model")
    tokenizer, model = load_reward_model(
        args.model,
        args.load_in_4bit,
        args.dtype,
        args.gpu_memory,
        args.cpu_memory,
    )
    print("  reward model loaded")

    print("[stage 5/5] Score rows and write checkpoints")
    score_dataframe(
        df=df,
        tokenizer=tokenizer,
        model=model,
        output_path=output_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        save_every=args.save_every,
        existing_rows=existing_rows,
    )


if __name__ == "__main__":
    main()
