import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import torch

from RM_LogisticRegression.models.calibration import read_joined_scores
from RM_LogisticRegression.models.mlp_calibration import train_mlp_calibrator
from RM_LogisticRegression.utils.paths import (
    default_calibrator_output_path,
    default_mlp_search_best_model_path,
    default_mlp_search_best_output_path,
    default_mlp_search_results_path,
)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search MLP calibrator hyperparameters on Gemma2 RM scores."
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
        "--results-output",
        default=None,
        help=(
            "Search summary CSV. Defaults to "
            "output/<run>/RM_LogisticRegression/rm_mlp_hparam_search_results.csv."
        ),
    )
    parser.add_argument(
        "--best-output",
        default=None,
        help=(
            "Best validation probability CSV. Defaults to "
            "output/<run>/RM_LogisticRegression/rm_mlp_hparam_best_valid_predictions.csv."
        ),
    )
    parser.add_argument(
        "--best-model-output",
        default=None,
        help=(
            "Best model checkpoint. Defaults to "
            "output/<run>/RM_LogisticRegression/rm_mlp_hparam_best_model.pt."
        ),
    )
    parser.add_argument(
        "--hidden-dims",
        default="16,32,64",
        help="Comma-separated hidden dimensions.",
    )
    parser.add_argument(
        "--dropouts",
        default="0.05,0.1,0.2",
        help="Comma-separated dropout values.",
    )
    parser.add_argument(
        "--learning-rates",
        default="0.0005,0.001",
        help="Comma-separated learning rates.",
    )
    parser.add_argument(
        "--weight-decays",
        default="0.001,0.003,0.01",
        help="Comma-separated AdamW weight decay values.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="512",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a CUDA device such as cuda:0.",
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
        help="Base random seed. Trial index is added to this value.",
    )
    parser.add_argument(
        "--same-seed",
        action="store_true",
        help="Use exactly --shuffle-seed for every trial instead of seed + trial index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[stage 1/5] Resolve paths and hyperparameter grid")
    results_output = (
        Path(args.results_output)
        if args.results_output is not None
        else default_mlp_search_results_path(args.output_date)
    )
    best_output = (
        Path(args.best_output)
        if args.best_output is not None
        else default_mlp_search_best_output_path(args.output_date)
    )
    best_model_output = (
        Path(args.best_model_output)
        if args.best_model_output is not None
        else default_mlp_search_best_model_path(args.output_date)
    )
    default_score_dir = default_calibrator_output_path(args.output_date).parent
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

    hidden_dims = parse_int_list(args.hidden_dims)
    dropouts = parse_float_list(args.dropouts)
    learning_rates = parse_float_list(args.learning_rates)
    weight_decays = parse_float_list(args.weight_decays)
    batch_sizes = parse_int_list(args.batch_sizes)
    grid = list(
        product(hidden_dims, dropouts, learning_rates, weight_decays, batch_sizes)
    )
    if not grid:
        raise ValueError("Hyperparameter grid is empty.")

    print(f"  train_labels: {args.train_labels}")
    print(f"  valid_labels: {args.valid_labels}")
    print(f"  train_scores: {train_scores}")
    print(f"  valid_scores: {valid_scores}")
    print(f"  results_output: {results_output}")
    print(f"  best_output: {best_output}")
    print(f"  best_model_output: {best_model_output}")
    print(f"  trials: {len(grid)}")
    print(f"  max_epochs: {args.max_epochs}")
    print(f"  patience: {args.patience}")
    print(f"  min_delta: {args.min_delta}")
    print(f"  device: {args.device}")

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

    print("[stage 4/5] Run hyperparameter search")
    records = []
    best_result = None
    best_record = None
    for trial_index, (
        hidden_dim,
        dropout,
        learning_rate,
        weight_decay,
        batch_size,
    ) in enumerate(grid, start=1):
        trial_seed = (
            args.shuffle_seed
            if args.same_seed
            else args.shuffle_seed + trial_index - 1
        )
        print(
            f"[trial {trial_index}/{len(grid)}] "
            f"hidden_dim={hidden_dim} dropout={dropout} "
            f"learning_rate={learning_rate} weight_decay={weight_decay} "
            f"batch_size={batch_size} seed={trial_seed}"
        )
        result = train_mlp_calibrator(
            train_df,
            valid_df,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            augment_swapped=args.augment_swapped,
            shuffle_seed=trial_seed,
            device_name=args.device,
        )
        record = {
            "rank": None,
            "trial": trial_index,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "seed": trial_seed,
            "best_epoch": result.checkpoint["best_epoch"],
            "best_valid_loss": result.checkpoint["best_valid_loss"],
            "log_loss": result.loss,
            "manual_log_loss": result.manual_loss,
            "accuracy": result.accuracy,
        }
        records.append(record)
        print(
            f"  trial_log_loss={result.loss:.8f} "
            f"accuracy={result.accuracy:.8f} "
            f"best_epoch={result.checkpoint['best_epoch']}"
        )

        if best_record is None or result.loss < best_record["log_loss"]:
            best_record = record
            best_result = result
            print(f"  new best log_loss={result.loss:.8f}")

    if best_result is None or best_record is None:
        raise RuntimeError("No search trial completed.")

    print("[stage 5/5] Write search results and best artifacts")
    results_df = pd.DataFrame(records).sort_values(
        ["log_loss", "accuracy"],
        ascending=[True, False],
    )
    results_df["rank"] = range(1, len(results_df) + 1)
    columns = [
        "rank",
        "trial",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "seed",
        "best_epoch",
        "best_valid_loss",
        "log_loss",
        "manual_log_loss",
        "accuracy",
    ]
    results_output.parent.mkdir(parents=True, exist_ok=True)
    results_df[columns].to_csv(results_output, index=False)
    best_sorted_record = results_df[columns].head(1).to_dict("records")[0]

    best_output.parent.mkdir(parents=True, exist_ok=True)
    best_result.output.to_csv(best_output, index=False)
    best_model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **best_result.checkpoint,
            "search_best_record": best_sorted_record,
            "train_labels": str(args.train_labels),
            "valid_labels": str(args.valid_labels),
            "train_scores": str(train_scores),
            "valid_scores": str(valid_scores),
            "allow_partial": args.allow_partial,
            "augment_swapped": args.augment_swapped,
            "same_seed": args.same_seed,
            "validation_log_loss": best_result.loss,
            "validation_accuracy": best_result.accuracy,
            "validation_rows": len(valid_df),
        },
        best_model_output,
    )

    print("Best trial:")
    print(results_df[columns].head(1).to_string(index=False))
    print(f"Wrote {results_output}")
    print(f"Wrote {best_output}")
    print(f"Wrote {best_model_output}")


if __name__ == "__main__":
    main()
