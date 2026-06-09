import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from RM_LogisticRegression.constants import LABEL_COLUMNS
from RM_LogisticRegression.data import read_required_columns, validate_one_hot_labels
from RM_LogisticRegression.metrics import prepare_prediction_probabilities


TEXT_COLUMNS = ["model_a", "model_b", "prompt", "response_a", "response_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract misclassified validation examples and build a second-round "
            "training CSV from errors plus sampled correct examples."
        )
    )
    parser.add_argument(
        "--labels",
        default="data/valid_split.csv",
        help="Validation CSV containing ids, text fields, and one-hot labels.",
    )
    parser.add_argument(
        "--predictions",
        default="output/gemma2_qlora_rm/gemma2_qlora_valid_predictions.csv",
        help="Prediction CSV containing id and probability columns.",
    )
    parser.add_argument(
        "--output",
        default="output/gemma2_qlora_rm/gemma2_qlora_error_examples.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--error-csv",
        default="output/gemma2_qlora_rm/gemma2_qlora_error_examples.csv",
        help="CSV output path for misclassified examples with diagnostic columns.",
    )
    parser.add_argument(
        "--round2-output",
        default="data/gemma2_qlora_rm_round2_train.csv",
        help="CSV output path for second-round training data.",
    )
    parser.add_argument(
        "--num-correct",
        default="same",
        help=(
            "Number of correctly classified examples to add to the round2 CSV. "
            "Use an integer, 'same' to match the number of errors, or 'all'."
        ),
    )
    parser.add_argument(
        "--correct-sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling correct examples.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each prediction row to sum to 1 before extracting errors.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow prediction files that contain only a subset of label ids.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of error examples to write. Default writes all errors.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["loss", "confidence", "id"],
        default="loss",
        help="Sort error examples by true-class loss, prediction confidence, or id.",
    )
    return parser.parse_args()


def as_label(row: pd.Series, suffix: str) -> str:
    return max(LABEL_COLUMNS, key=lambda column: row[f"{column}_{suffix}"])


def json_list_to_markdown(value: object) -> str:
    if not isinstance(value, str) or value == "":
        return ""

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return str(value).strip()

    if not isinstance(parsed, list):
        return str(parsed).strip()

    return "\n\n".join("" if item is None else str(item).strip() for item in parsed)


def fenced_block(value: object) -> str:
    text = json_list_to_markdown(value)
    return f"```text\n{text}\n```"


def probability_line(row: dict[str, object]) -> str:
    return ", ".join(
        f"{column}={row[f'{column}_pred']:.6f}" for column in LABEL_COLUMNS
    )


def true_probability(row: pd.Series) -> float:
    true_label = row["true_label"]
    return float(row[f"{true_label}_pred"])


def parse_num_correct(value: str, error_count: int, correct_count: int) -> int:
    normalized = value.strip().lower()
    if normalized == "same":
        return min(error_count, correct_count)
    if normalized == "all":
        return correct_count
    try:
        count = int(normalized)
    except ValueError as exc:
        raise ValueError(
            "--num-correct must be an integer, 'same', or 'all'."
        ) from exc
    if count < 0:
        raise ValueError("--num-correct cannot be negative.")
    return min(count, correct_count)


def sample_correct_examples(
    correct: pd.DataFrame,
    requested: int,
    seed: int,
) -> pd.DataFrame:
    if requested == 0:
        return correct.head(0).copy()
    return correct.sample(n=requested, random_state=seed).copy()


def to_training_rows(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["id", *TEXT_COLUMNS, *[f"{column}_true" for column in LABEL_COLUMNS]]
    training_rows = df[columns].copy()
    training_rows = training_rows.rename(
        columns={f"{column}_true": column for column in LABEL_COLUMNS}
    )
    return training_rows[["id", *TEXT_COLUMNS, *LABEL_COLUMNS]]


def write_csv_outputs(
    errors: pd.DataFrame,
    correct: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[int, Path, Path]:
    error_csv_path = Path(args.error_csv)
    round2_path = Path(args.round2_output)
    error_csv_path.parent.mkdir(parents=True, exist_ok=True)
    round2_path.parent.mkdir(parents=True, exist_ok=True)

    errors.to_csv(error_csv_path, index=False)

    correct_count = parse_num_correct(args.num_correct, len(errors), len(correct))
    sampled_correct = sample_correct_examples(
        correct,
        requested=correct_count,
        seed=args.correct_sample_seed,
    )
    round2 = pd.concat(
        [
            to_training_rows(errors),
            to_training_rows(sampled_correct),
        ],
        ignore_index=True,
    )
    round2 = round2.sample(frac=1.0, random_state=args.correct_sample_seed).reset_index(
        drop=True
    )
    round2.to_csv(round2_path, index=False)
    return correct_count, error_csv_path, round2_path


def write_report(errors: pd.DataFrame, output_path: Path, total_rows: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Prediction Error Examples",
        "",
        f"- Total matched rows: {total_rows}",
        f"- Error rows written: {len(errors)}",
    ]
    if total_rows:
        lines.append(f"- Error rate: {len(errors) / total_rows:.4f}")
    lines.append("")

    for index, row in enumerate(errors.itertuples(index=False), start=1):
        row_dict = row._asdict()
        lines.extend(
            [
                f"## {index}. id={row_dict['id']}",
                "",
                f"- model_a: `{row_dict.get('model_a', '')}`",
                f"- model_b: `{row_dict.get('model_b', '')}`",
                f"- true_label: `{row_dict['true_label']}`",
                f"- pred_label: `{row_dict['pred_label']}`",
                f"- true_probability: {row_dict['true_probability']:.6f}",
                f"- pred_confidence: {row_dict['pred_confidence']:.6f}",
                f"- true_class_loss: {row_dict['true_class_loss']:.6f}",
                f"- probabilities: {probability_line(row_dict)}",
                "",
                "### Prompt",
                "",
                fenced_block(row_dict.get("prompt", "")),
                "",
                "### Response A",
                "",
                fenced_block(row_dict.get("response_a", "")),
                "",
                "### Response B",
                "",
                fenced_block(row_dict.get("response_b", "")),
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    label_path = Path(args.labels)
    prediction_path = Path(args.predictions)
    output_path = Path(args.output)

    labels = read_required_columns(label_path, ["id", *TEXT_COLUMNS, *LABEL_COLUMNS])
    predictions = read_required_columns(prediction_path, ["id", *LABEL_COLUMNS])
    validate_one_hot_labels(labels, label_path)

    merged = labels[["id", *TEXT_COLUMNS, *LABEL_COLUMNS]].merge(
        predictions[["id", *LABEL_COLUMNS]],
        on="id",
        suffixes=("_true", "_pred"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(labels) and not args.allow_partial:
        raise ValueError(
            f"Only matched {len(merged)} of {len(labels)} label rows by id."
        )
    if len(merged) == 0:
        raise ValueError("No prediction rows matched label ids.")

    pred_cols = [f"{column}_pred" for column in LABEL_COLUMNS]
    prepared_predictions = prepare_prediction_probabilities(
        merged.rename(
            columns={column: column.removesuffix("_pred") for column in pred_cols}
        ),
        normalize=args.normalize,
    )
    merged.loc[:, pred_cols] = prepared_predictions

    merged["true_label"] = merged.apply(as_label, axis=1, suffix="true")
    merged["pred_label"] = merged.apply(as_label, axis=1, suffix="pred")
    merged["true_probability"] = merged.apply(true_probability, axis=1)
    merged["pred_confidence"] = merged[pred_cols].max(axis=1)
    merged["true_class_loss"] = -merged["true_probability"].map(
        lambda probability: math.log(max(float(probability), 1e-15))
    )

    errors = merged[merged["true_label"] != merged["pred_label"]].copy()
    correct = merged[merged["true_label"] == merged["pred_label"]].copy()
    ascending = args.sort_by == "id"
    sort_column = {
        "loss": "true_class_loss",
        "confidence": "pred_confidence",
        "id": "id",
    }[args.sort_by]
    errors = errors.sort_values(sort_column, ascending=ascending)
    if args.limit is not None:
        errors = errors.head(args.limit)

    correct_count, error_csv_path, round2_path = write_csv_outputs(
        errors,
        correct,
        args,
    )
    write_report(errors, output_path, total_rows=len(merged))
    print(f"matched_rows={len(merged)}")
    print(f"errors_written={len(errors)}")
    print(f"correct_rows_available={len(correct)}")
    print(f"correct_rows_added={correct_count}")
    print(f"wrote_error_csv={error_csv_path}")
    print(f"wrote_round2_train={round2_path}")
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
