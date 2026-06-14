import argparse
from pathlib import Path

import joblib

from RM_LogisticRegression.models.calibration import read_joined_scores, train_logistic_calibrator
from RM_LogisticRegression.utils.constants import LABEL_COLUMNS, RM_FEATURE_COLUMNS
from RM_LogisticRegression.utils.paths import (
    default_calibrator_model_path,
    default_calibrator_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a three-class calibrator on Gemma2 RM scores."
    )
    parser.add_argument(
        "--train-labels",
        default="data/train_split.csv",
        help="Training split with one-hot labels.",
    )
    parser.add_argument(
        "--valid-labels",
        default="data/valid_split.csv",
        help="Validation split with one-hot labels.",
    )
    parser.add_argument(
        "--train-scores",
        default=None,
        help=(
            "Gemma RM scores for the training split. Defaults to the selected "
            "run folder's train score file."
        ),
    )
    parser.add_argument(
        "--valid-scores",
        default=None,
        help=(
            "Gemma RM scores for the validation split. Defaults to the selected "
            "run folder's valid score file."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Validation probability output CSV. Defaults to "
            "output/<run>/RM_LogisticRegression/rm_calibrated_valid_predictions.csv."
        ),
    )
    parser.add_argument(
        "--model-output",
        default=None,
        help=(
            "Trained classifier output path. Defaults to "
            "output/<run>/RM_LogisticRegression/rm_logistic_regression_model.joblib."
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
        "--c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum optimizer iterations.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Evaluate on rows that have scores instead of requiring full split coverage.",
    )
    parser.add_argument(
        "--augment-swapped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Augment training data by swapping response A/B features and labels. "
            "Enabled by default."
        ),
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed used to shuffle augmented training rows.",
    )
    parser.add_argument(
        "--valid-tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Average validation predictions with swapped A/B test-time augmentation. "
            "Enabled by default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[stage 1/5] Resolve paths and configuration")
    default_output_path = default_calibrator_output_path(args.output_date)
    default_model_path = default_calibrator_model_path(args.output_date)
    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path
    )
    model_output_path = (
        Path(args.model_output)
        if args.model_output is not None
        else default_model_path
    )
    default_score_dir = default_output_path.parent
    train_scores = (
        Path(args.train_scores)
        if args.train_scores is not None
        else default_score_dir / "train_split_gemma_rm_scores.csv"
    )
    valid_scores = (
        Path(args.valid_scores)
        if args.valid_scores is not None
        else default_score_dir / "valid_split_gemma_rm_scores.csv"
    )
    print(f"  train_labels: {args.train_labels}")
    print(f"  valid_labels: {args.valid_labels}")
    print(f"  train_scores: {train_scores}")
    print(f"  valid_scores: {valid_scores}")
    print(f"  output: {output_path}")
    print(f"  model_output: {model_output_path}")
    print(f"  allow_partial: {args.allow_partial}")
    print(f"  augment_swapped: {args.augment_swapped}")
    print(f"  shuffle_seed: {args.shuffle_seed}")
    print(f"  valid_tta: {args.valid_tta}")
    print(f"  logistic_regression_C: {args.c}")
    print(f"  max_iter: {args.max_iter}")

    print("[stage 2/5] Read and join train score features")
    train_df = read_joined_scores(
        Path(args.train_labels),
        train_scores,
        args.allow_partial,
    )
    print(f"  train rows: {len(train_df)}")

    print("[stage 3/5] Read and join validation score features")
    valid_df = read_joined_scores(
        Path(args.valid_labels),
        valid_scores,
        args.allow_partial,
    )
    print(f"  valid rows: {len(valid_df)}")

    print("[stage 4/5] Train LogisticRegression calibrator and predict validation probabilities")
    result = train_logistic_calibrator(
        train_df,
        valid_df,
        c=args.c,
        max_iter=args.max_iter,
        augment_swapped=args.augment_swapped,
        shuffle_seed=args.shuffle_seed,
        valid_tta=args.valid_tta,
    )

    print("[stage 5/5] Write predictions and report log loss")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.output.to_csv(output_path, index=False)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": result.model,
            "feature_columns": RM_FEATURE_COLUMNS,
            "label_columns": LABEL_COLUMNS,
            "train_labels": str(args.train_labels),
            "valid_labels": str(args.valid_labels),
            "train_scores": str(train_scores),
            "valid_scores": str(valid_scores),
            "c": args.c,
            "max_iter": args.max_iter,
            "allow_partial": args.allow_partial,
            "augment_swapped": args.augment_swapped,
            "shuffle_seed": args.shuffle_seed,
            "valid_tta": args.valid_tta,
            "validation_log_loss": result.loss,
            "validation_accuracy": result.accuracy,
            "validation_base_log_loss": result.base_loss,
            "validation_base_accuracy": result.base_accuracy,
            "validation_tta_log_loss": result.tta_loss,
            "validation_tta_accuracy": result.tta_accuracy,
            "validation_rows": len(valid_df),
        },
        model_output_path,
    )

    print(f"base_log_loss={result.base_loss:.8f}")
    print(f"base_manual_log_loss={result.base_manual_loss:.8f}")
    print(f"base_accuracy={result.base_accuracy:.8f}")
    if result.tta_loss is not None:
        print(f"tta_log_loss={result.tta_loss:.8f}")
        print(f"tta_manual_log_loss={result.tta_manual_loss:.8f}")
        print(f"tta_accuracy={result.tta_accuracy:.8f}")
    print(f"log_loss={result.loss:.8f}")
    print(f"manual_log_loss={result.manual_loss:.8f}")
    print(f"accuracy={result.accuracy:.8f}")
    print(f"rows={len(valid_df)}")
    print(f"Wrote {output_path}")
    print(f"Wrote {model_output_path}")


if __name__ == "__main__":
    main()
