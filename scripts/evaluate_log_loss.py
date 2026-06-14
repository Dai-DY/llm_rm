import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from RM_LogisticRegression.utils.constants import LABEL_COLUMNS
from RM_LogisticRegression.data.processing import read_required_columns, validate_one_hot_labels
from RM_LogisticRegression.utils.metrics import (
    multiclass_accuracy,
    multiclass_log_loss,
    prepare_prediction_probabilities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction probabilities with multi-class log loss."
    )
    parser.add_argument(
        "--labels",
        default="data/valid_split.csv",
        help="CSV containing id and one-hot ground truth label columns.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="CSV containing id and predicted probability columns.",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1e-15,
        help="Probability clipping epsilon used before log loss.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each prediction row to sum to 1 before evaluation.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Evaluate matched ids only instead of requiring all label rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_path = Path(args.labels)
    prediction_path = Path(args.predictions)

    labels = read_required_columns(label_path, ["id", *LABEL_COLUMNS])
    predictions = read_required_columns(prediction_path, ["id", *LABEL_COLUMNS])
    validate_one_hot_labels(labels, label_path)

    merged = labels[["id", *LABEL_COLUMNS]].merge(
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

    y_true = merged[[f"{col}_true" for col in LABEL_COLUMNS]].to_numpy()
    pred_cols = [f"{col}_pred" for col in LABEL_COLUMNS]
    y_pred = prepare_prediction_probabilities(
        merged.rename(columns={col: col.removesuffix("_pred") for col in pred_cols}),
        normalize=args.normalize,
    )

    loss = multiclass_log_loss(y_true, y_pred, args.clip)
    accuracy = multiclass_accuracy(y_true, y_pred)
    print(f"log_loss={loss:.8f}")
    print(f"accuracy={accuracy:.8f}")
    print(f"rows={len(merged)}")


if __name__ == "__main__":
    main()
