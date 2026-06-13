from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "src" / "Gemma2_QLoRA" / "train_gemma2_mlp_head_from_features.py"


@dataclass(frozen=True)
class SearchResult:
    head_dropout: float
    head_hidden_ratio: float
    learning_rate: float
    weight_decay: float
    output_dir: str
    checkpoint: str
    predictions: str
    valid_log_loss: float | None
    valid_accuracy: float | None
    best_epoch: int | None
    status: str
    error: str | None = None


def parse_float_grid(value: str) -> list[float]:
    values = []
    for raw in value.split(","):
        raw = raw.strip()
        if raw:
            values.append(float(raw))
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one float.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search MLP head hyperparameters from saved Gemma2 pooled features."
    )
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--valid-features", required=True)
    parser.add_argument(
        "--search-output-dir",
        default="output/gemma2_feature_mlp_head_search",
        help="Directory containing one run per hyperparameter combination.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Defaults to <search-output-dir>/gemma2_feature_mlp_head_search_results.csv.",
    )
    parser.add_argument(
        "--dropouts",
        type=parse_float_grid,
        default=parse_float_grid("0.0,0.05,0.1,0.15,0.2"),
        help="Comma-separated dropout grid.",
    )
    parser.add_argument(
        "--hidden-ratios",
        type=parse_float_grid,
        default=parse_float_grid("0.25,0.5,0.75,1.0"),
        help="Comma-separated hidden-ratio grid.",
    )
    parser.add_argument(
        "--learning-rates",
        type=parse_float_grid,
        default=parse_float_grid("3e-4,1e-3,3e-3"),
        help="Comma-separated learning-rate grid.",
    )
    parser.add_argument(
        "--weight-decays",
        type=parse_float_grid,
        default=parse_float_grid("0.0,0.01"),
        help="Comma-separated weight-decay grid.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def format_float_for_path(value: float) -> str:
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


def run_name(dropout: float, ratio: float, learning_rate: float, weight_decay: float) -> str:
    return (
        f"dropout{format_float_for_path(dropout)}_"
        f"ratio{format_float_for_path(ratio)}_"
        f"lr{format_float_for_path(learning_rate)}_"
        f"wd{format_float_for_path(weight_decay)}"
    )


def build_command(
    args: argparse.Namespace,
    output_dir: Path,
    dropout: float,
    ratio: float,
    learning_rate: float,
    weight_decay: float,
) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--train-features",
        args.train_features,
        "--valid-features",
        args.valid_features,
        "--output-dir",
        str(output_dir),
        "--head-dropout",
        str(dropout),
        "--head-hidden-ratio",
        str(ratio),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--patience",
        str(args.patience),
        "--min-delta",
        str(args.min_delta),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def read_metrics(metrics_path: Path) -> tuple[float, float, int]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return (
        float(metrics["valid_log_loss"]),
        float(metrics["valid_accuracy"]),
        int(metrics["best_epoch"]),
    )


def write_results(path: Path, results: list[SearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "head_dropout",
        "head_hidden_ratio",
        "learning_rate",
        "weight_decay",
        "valid_log_loss",
        "valid_accuracy",
        "best_epoch",
        "status",
        "error",
        "output_dir",
        "checkpoint",
        "predictions",
    ]
    ordered = sorted(
        results,
        key=lambda item: item.valid_log_loss
        if item.valid_log_loss is not None
        else float("inf"),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(ordered, start=1):
            row = result.__dict__.copy()
            row["rank"] = rank
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    search_output_dir = Path(args.search_output_dir)
    results_csv = (
        Path(args.results_csv)
        if args.results_csv is not None
        else search_output_dir / "gemma2_feature_mlp_head_search_results.csv"
    )
    results: list[SearchResult] = []
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    scripts_path = str(ROOT / "scripts")
    env["PYTHONPATH"] = os.pathsep.join(
        [src_path, scripts_path, env.get("PYTHONPATH", "")]
    )

    for dropout in args.dropouts:
        for ratio in args.hidden_ratios:
            for learning_rate in args.learning_rates:
                for weight_decay in args.weight_decays:
                    output_dir = search_output_dir / run_name(
                        dropout,
                        ratio,
                        learning_rate,
                        weight_decay,
                    )
                    checkpoint_path = output_dir / "gemma2_mlp_head.pt"
                    predictions_path = output_dir / "gemma2_mlp_head_valid_predictions.csv"
                    metrics_path = output_dir / "gemma2_mlp_head_metrics.json"
                    command = build_command(
                        args,
                        output_dir,
                        dropout,
                        ratio,
                        learning_rate,
                        weight_decay,
                    )

                    if args.dry_run:
                        print(" ".join(command))
                        continue

                    if args.force or not metrics_path.exists():
                        output_dir.mkdir(parents=True, exist_ok=True)
                        print(
                            "[search] "
                            f"dropout={dropout} ratio={ratio} "
                            f"lr={learning_rate} wd={weight_decay}"
                        )
                        try:
                            subprocess.run(command, check=True, cwd=ROOT, env=env)
                        except subprocess.CalledProcessError as exc:
                            results.append(
                                SearchResult(
                                    head_dropout=dropout,
                                    head_hidden_ratio=ratio,
                                    learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    output_dir=str(output_dir),
                                    checkpoint=str(checkpoint_path),
                                    predictions=str(predictions_path),
                                    valid_log_loss=None,
                                    valid_accuracy=None,
                                    best_epoch=None,
                                    status="failed",
                                    error=f"train script exited with code {exc.returncode}",
                                )
                            )
                            write_results(results_csv, results)
                            continue
                    else:
                        print(f"[search] reusing {metrics_path}")

                    valid_log_loss, valid_accuracy, best_epoch = read_metrics(metrics_path)
                    results.append(
                        SearchResult(
                            head_dropout=dropout,
                            head_hidden_ratio=ratio,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            output_dir=str(output_dir),
                            checkpoint=str(checkpoint_path),
                            predictions=str(predictions_path),
                            valid_log_loss=valid_log_loss,
                            valid_accuracy=valid_accuracy,
                            best_epoch=best_epoch,
                            status="ok",
                        )
                    )
                    write_results(results_csv, results)

    write_results(results_csv, results)
    if results:
        best = min(
            results,
            key=lambda item: item.valid_log_loss
            if item.valid_log_loss is not None
            else float("inf"),
        )
        print(
            "best: "
            f"dropout={best.head_dropout} "
            f"ratio={best.head_hidden_ratio} "
            f"lr={best.learning_rate} "
            f"wd={best.weight_decay} "
            f"log_loss={best.valid_log_loss}"
        )
        print(f"checkpoint: {best.checkpoint}")
        print(f"predictions: {best.predictions}")
    print(f"results: {results_csv}")


if __name__ == "__main__":
    main()
