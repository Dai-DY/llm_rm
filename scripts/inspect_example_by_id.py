import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from RM_LogisticRegression.utils.constants import LABEL_COLUMNS
from RM_LogisticRegression.data.processing import read_required_columns, validate_one_hot_labels


TEXT_COLUMNS = ["model_a", "model_b", "prompt", "response_a", "response_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively inspect validation examples by id."
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help="Optional ids to print immediately. If omitted, starts interactive mode.",
    )
    parser.add_argument(
        "--labels",
        default="data/valid_split.csv",
        help="Validation CSV containing ids, text fields, and one-hot labels.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional prediction CSV. If provided, prediction probabilities are shown.",
    )
    return parser.parse_args()


def parse_json_list(value: object) -> list[str]:
    if not isinstance(value, str) or value == "":
        return [""]

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value.strip()]

    if not isinstance(parsed, list):
        return [str(parsed).strip()]

    return ["" if item is None else str(item).strip() for item in parsed]


def format_text(value: object) -> str:
    return "\n\n".join(parse_json_list(value)).strip()


def true_label(row: pd.Series) -> str:
    return max(LABEL_COLUMNS, key=lambda column: row[column])


def label_marker(label: str, response_label: str) -> str:
    return " [TRUE]" if label == response_label else ""


def load_predictions(path_value: str | None) -> pd.DataFrame | None:
    if path_value is None:
        return None

    path = Path(path_value)
    predictions = read_required_columns(path, ["id", *LABEL_COLUMNS])
    return predictions.set_index("id", drop=False)


def print_example(
    row: pd.Series,
    predictions: pd.DataFrame | None = None,
) -> None:
    label = true_label(row)
    print("\n" + "=" * 88)
    print(f"id: {row['id']}")
    print(f"model_a: {row['model_a']}")
    print(f"model_b: {row['model_b']}")
    print(f"true_label: {label}")

    if predictions is not None and row["id"] in predictions.index:
        pred_row = predictions.loc[row["id"]]
        pred_label = max(LABEL_COLUMNS, key=lambda column: pred_row[column])
        probabilities = ", ".join(
            f"{column}={float(pred_row[column]):.6f}" for column in LABEL_COLUMNS
        )
        print(f"pred_label: {pred_label}")
        print(f"probabilities: {probabilities}")

    print("-" * 88)
    print("PROMPT")
    print(format_text(row["prompt"]))
    print("-" * 88)
    print(f"RESPONSE A - {row['model_a']}{label_marker(label, 'winner_model_a')}")
    print(format_text(row["response_a"]))
    print("-" * 88)
    print(f"RESPONSE B - {row['model_b']}{label_marker(label, 'winner_model_b')}")
    print(format_text(row["response_b"]))
    if label == "winner_tie":
        print("-" * 88)
        print("TRUE RESULT: TIE")
    print("=" * 88 + "\n")


def print_by_id(
    id_value: str,
    examples: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
) -> None:
    try:
        numeric_id = int(id_value)
    except ValueError:
        print(f"Invalid id: {id_value!r}")
        return

    if numeric_id not in examples.index:
        print(f"id not found: {numeric_id}")
        return

    print_example(examples.loc[numeric_id], predictions)


def interactive_loop(
    examples: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
) -> None:
    print("Input an id to inspect. Use q/quit/exit to stop.")
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
        print_by_id(id_value, examples, predictions)


def main() -> None:
    args = parse_args()
    label_path = Path(args.labels)

    examples = read_required_columns(label_path, ["id", *TEXT_COLUMNS, *LABEL_COLUMNS])
    validate_one_hot_labels(examples, label_path)
    examples = examples.set_index("id", drop=False)
    predictions = load_predictions(args.predictions)

    if args.ids:
        for id_value in args.ids:
            print_by_id(id_value, examples, predictions)
    else:
        interactive_loop(examples, predictions)


if __name__ == "__main__":
    main()
