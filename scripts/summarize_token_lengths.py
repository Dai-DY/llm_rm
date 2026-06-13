import argparse
from pathlib import Path

import pandas as pd

from token_length_by_id import (
    TEXT_COLUMNS,
    build_text,
    default_model_path,
    load_tokenizer,
)


PERCENTILES = [0.5, 0.75, 0.9, 0.95, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize token lengths for a preference CSV after text concatenation."
    )
    parser.add_argument(
        "--input",
        default="data/train_split.csv",
        help="CSV containing id, prompt, response_a, and response_b columns.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Tokenizer path. Defaults to the local Gemma2 model path.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1800,
        help="Training max length used to report truncation counts.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Tokenizer batch size.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Print the longest k rows by token length.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for per-row id/token_length/truncated results.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> pd.DataFrame:
    rows = pd.read_csv(csv_path, dtype={"id": str})
    missing = [column for column in TEXT_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    return rows


def batched_token_lengths(
    tokenizer,
    rows: pd.DataFrame,
    batch_size: int,
) -> list[int]:
    lengths: list[int] = []
    for start in range(0, len(rows), batch_size):
        batch = rows.iloc[start : start + batch_size]
        texts = [build_text(row) for _, row in batch.iterrows()]
        encoded = tokenizer(
            texts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return lengths


def print_summary(
    result: pd.DataFrame,
    csv_path: Path,
    model_path: str,
    max_length: int,
    top_k: int,
) -> None:
    lengths = result["token_length"]
    truncated_count = int(result["truncated"].sum())
    truncated_rate = truncated_count / len(result) if len(result) else 0.0

    print("\nToken length summary")
    print("=" * 88)
    print(f"input: {csv_path}")
    print("format: gemma2")
    print(f"tokenizer: {model_path}")
    print(f"rows: {len(result)}")
    print(f"max_length: {max_length}")
    print(f"truncated_rows: {truncated_count} ({truncated_rate:.2%})")
    print(f"min: {int(lengths.min())}")
    print(f"mean: {lengths.mean():.2f}")
    print(f"max: {int(lengths.max())}")
    for percentile in PERCENTILES:
        label = int(percentile * 100)
        value = lengths.quantile(percentile)
        print(f"p{label}: {value:.0f}")

    if top_k <= 0:
        return

    print("-" * 88)
    print(f"longest_{top_k}:")
    longest = result.sort_values("token_length", ascending=False).head(top_k)
    for _, row in longest.iterrows():
        print(
            f"id={row['id']} token_length={int(row['token_length'])} "
            f"truncated={'yes' if row['truncated'] else 'no'}"
        )
    print("=" * 88)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)
    model_path = args.model or default_model_path()

    print(f"Loading tokenizer: {model_path}")
    tokenizer = load_tokenizer(model_path)

    rows = read_rows(csv_path)
    lengths = batched_token_lengths(
        tokenizer=tokenizer,
        rows=rows,
        batch_size=args.batch_size,
    )

    result = pd.DataFrame(
        {
            "id": rows["id"].astype(str),
            "token_length": lengths,
        }
    )
    result["truncated"] = result["token_length"] > args.max_length

    print_summary(
        result=result,
        csv_path=csv_path,
        model_path=model_path,
        max_length=args.max_length,
        top_k=args.top_k,
    )

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
