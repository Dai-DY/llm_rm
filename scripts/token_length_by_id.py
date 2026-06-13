import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from Gemma2_QLoRA.constants import DEFAULT_MODEL_PATH as DEFAULT_GEMMA2_MODEL_PATH
from Gemma2_QLoRA.data import build_compact_pair_text


TEXT_COLUMNS = ["id", "prompt", "response_a", "response_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load examples by id and print token lengths after text concatenation."
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help="Example ids to inspect. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--input",
        default="data/valid_split.csv",
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
        default=None,
        help="Also print token length after truncation to this max length.",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print the concatenated text before tokenizing.",
    )
    return parser.parse_args()


def default_model_path() -> str:
    return DEFAULT_GEMMA2_MODEL_PATH


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


def build_text(row: pd.Series) -> str:
    return build_compact_pair_text(row["prompt"], row["response_a"], row["response_b"])


def read_examples(csv_path: Path) -> pd.DataFrame:
    examples = pd.read_csv(csv_path, dtype={"id": str})
    missing = [column for column in TEXT_COLUMNS if column not in examples.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    return examples.set_index("id", drop=False)


def token_count(tokenizer, text: str, max_length: int | None = None) -> int:
    encoded = tokenizer(
        text,
        truncation=max_length is not None,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    return len(encoded["input_ids"])


def print_token_length(
    id_value: str,
    examples: pd.DataFrame,
    tokenizer,
    max_length: int | None,
    show_text: bool,
) -> None:
    if id_value not in examples.index:
        print(f"id not found: {id_value}")
        return

    row = examples.loc[id_value]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    text = build_text(row)
    full_length = token_count(tokenizer, text)
    print("\n" + "=" * 88)
    print(f"id: {id_value}")
    print("format: gemma2")
    print(f"token_length: {full_length}")
    if max_length is not None:
        truncated_length = token_count(tokenizer, text, max_length=max_length)
        truncated = "yes" if full_length > max_length else "no"
        print(f"max_length: {max_length}")
        print(f"truncated: {truncated}")
        print(f"token_length_after_truncation: {truncated_length}")
    if show_text:
        print("-" * 88)
        print(text)
    print("=" * 88)


def interactive_loop(
    examples: pd.DataFrame,
    tokenizer,
    max_length: int | None,
    show_text: bool,
) -> None:
    print("Input an id to inspect token length. Use q/quit/exit to stop.")
    while True:
        try:
            id_value = input("id> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if id_value.lower() in {"q", "quit", "exit"}:
            break
        if not id_value:
            continue
        print_token_length(
            id_value=id_value,
            examples=examples,
            tokenizer=tokenizer,
            max_length=max_length,
            show_text=show_text,
        )


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)
    model_path = args.model or default_model_path()

    print(f"Loading tokenizer: {model_path}")
    tokenizer = load_tokenizer(model_path)
    examples = read_examples(csv_path)

    if args.ids:
        for id_value in args.ids:
            print_token_length(
                id_value=str(id_value),
                examples=examples,
                tokenizer=tokenizer,
                max_length=args.max_length,
                show_text=args.show_text,
            )
    else:
        interactive_loop(
            examples=examples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            show_text=args.show_text,
        )


if __name__ == "__main__":
    main()
