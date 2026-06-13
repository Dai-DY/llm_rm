from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from Gemma2_QLoRA.constants import LABEL_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "src" / "Gemma2_QLoRA" / "train.py"
GEMMA2_LINEAR_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(frozen=True)
class SearchResult:
    head_dropout: float
    head_hidden_ratio: float
    output_dir: str
    predictions: str
    log_loss: float | None
    accuracy: float | None
    rows: int | None
    best_trainer_metric: float | None
    best_global_step: int | None
    status: str
    error: str | None = None


def parse_float_grid(value: str) -> list[float]:
    values = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(float(raw))
    if not values:
        raise argparse.ArgumentTypeError("Grid must contain at least one float.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search Gemma2 QLoRA MLP classification-head hyperparameters."
    )
    parser.add_argument("--model", default="models/sfairXC__FsfairX-Gemma2-RM-v0.1")
    parser.add_argument("--train", default="data/train_split.csv")
    parser.add_argument("--valid", default="data/valid_split.csv")
    parser.add_argument(
        "--search-output-dir",
        default="output/gemma2_qlora_rm_rep_head_search",
        help="Directory containing one training run per hyperparameter combination.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Defaults to <search-output-dir>/gemma_mlp_head_hparam_search_results.csv.",
    )
    parser.add_argument(
        "--baseline-predictions",
        default="output/gemma2_qlora_rm_rep/gemma2_qlora_valid_predictions.csv",
        help="Optional existing predictions to include as the current baseline.",
    )
    parser.add_argument(
        "--baseline-config",
        default="output/gemma2_qlora_rm_rep/adapter/gemma2_qlora_config.json",
        help="Optional existing config for baseline head hyperparameters.",
    )
    parser.add_argument(
        "--resume-adapter",
        default=None,
        help="Optional adapter to initialize each run from before searching the MLP head.",
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
    parser.add_argument("--hardware-profile", default="auto")
    parser.add_argument("--max-length", type=int, default=1800)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--head-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the base model and LoRA weights; train only the MLP head.",
    )
    parser.add_argument(
        "--target-modules",
        default=",".join(GEMMA2_LINEAR_TARGET_MODULES),
        help="Comma-separated LoRA targets.",
    )
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-valid", type=int, default=None)
    parser.add_argument("--swap-consistency-weight", type=float, default=0.05)
    parser.add_argument("--swap-ce-weight", type=float, default=0.0)
    parser.add_argument("--prototype-loss-weight", type=float, default=0.02)
    parser.add_argument("--prototype-momentum", type=float, default=0.95)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--disable-softcapping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--valid-tta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swap-augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun combinations even when validation predictions already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the training commands without running them.",
    )
    return parser.parse_args()


def label_index(row: dict[str, str]) -> int:
    values = [float(row[column]) for column in LABEL_COLUMNS]
    if sum(value == 1.0 for value in values) != 1:
        raise ValueError(f"Expected one-hot labels for id={row.get('id')}.")
    return values.index(1.0)


def read_labels(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", *LABEL_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        return {row["id"]: label_index(row) for row in reader}


def evaluate_predictions(label_by_id: dict[str, int], predictions_path: Path) -> tuple[float, float, int]:
    total_loss = 0.0
    correct = 0
    rows = 0
    with predictions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id", *LABEL_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{predictions_path} missing columns: {sorted(missing)}")
        for row in reader:
            row_id = row["id"]
            if row_id not in label_by_id:
                continue
            probs = [max(1e-15, min(1.0 - 1e-15, float(row[column]))) for column in LABEL_COLUMNS]
            prob_sum = sum(probs)
            probs = [prob / prob_sum for prob in probs]
            target = label_by_id[row_id]
            total_loss -= math.log(probs[target])
            correct += int(max(range(len(probs)), key=probs.__getitem__) == target)
            rows += 1
    if rows != len(label_by_id):
        raise ValueError(
            f"{predictions_path} matched {rows} of {len(label_by_id)} validation rows."
        )
    return total_loss / rows, correct / rows, rows


def trainer_state_metrics(output_dir: Path) -> tuple[float | None, int | None]:
    candidates = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    state_path = output_dir / "trainer_state.json"
    if state_path.exists():
        candidates.append(state_path)
    if not candidates:
        return None, None
    state = json.loads(candidates[-1].read_text(encoding="utf-8"))
    best_metric = state.get("best_metric")
    best_step = state.get("best_global_step")
    return (
        float(best_metric) if best_metric is not None else None,
        int(best_step) if best_step is not None else None,
    )


def format_float_for_path(value: float) -> str:
    return f"{value:.4g}".replace("-", "m").replace(".", "p")


def run_name(dropout: float, hidden_ratio: float) -> str:
    return (
        f"dropout{format_float_for_path(dropout)}_"
        f"ratio{format_float_for_path(hidden_ratio)}"
    )


def optional_arg(command: list[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def bool_arg(command: list[str], name: str, value: bool | None) -> None:
    if value is None:
        return
    command.append(name if value else f"--no-{name.removeprefix('--')}")


def build_train_command(args: argparse.Namespace, output_dir: Path, dropout: float, ratio: float) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        args.model,
        "--train",
        args.train,
        "--valid",
        args.valid,
        "--output-dir",
        str(output_dir),
        "--hardware-profile",
        args.hardware_profile,
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--target-modules",
        args.target_modules,
        "--classifier-head",
        "mlp",
        "--head-dropout",
        str(dropout),
        "--head-hidden-ratio",
        str(ratio),
        "--seed",
        str(args.seed),
        "--logging-steps",
        str(args.logging_steps),
        "--eval-steps",
        str(args.eval_steps),
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--swap-consistency-weight",
        str(args.swap_consistency_weight),
        "--swap-ce-weight",
        str(args.swap_ce_weight),
        "--prototype-loss-weight",
        str(args.prototype_loss_weight),
        "--prototype-momentum",
        str(args.prototype_momentum),
    ]
    optional_arg(command, "--resume-adapter", args.resume_adapter)
    optional_arg(command, "--batch-size", args.batch_size)
    optional_arg(command, "--eval-batch-size", args.eval_batch_size)
    optional_arg(command, "--gradient-accumulation-steps", args.gradient_accumulation_steps)
    optional_arg(command, "--dtype", args.dtype)
    optional_arg(command, "--limit-train", args.limit_train)
    optional_arg(command, "--limit-valid", args.limit_valid)
    bool_arg(command, "--load-in-4bit", args.load_in_4bit)
    bool_arg(command, "--gradient-checkpointing", args.gradient_checkpointing)
    bool_arg(command, "--disable-softcapping", args.disable_softcapping)
    bool_arg(command, "--valid-tta", args.valid_tta)
    bool_arg(command, "--swap-augmentation", args.swap_augmentation)
    bool_arg(command, "--head-only", args.head_only)
    return command


def include_baseline(args: argparse.Namespace, label_by_id: dict[str, int]) -> SearchResult | None:
    predictions_path = Path(args.baseline_predictions)
    config_path = Path(args.baseline_config)
    if not predictions_path.exists() or not config_path.exists():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    loss, accuracy, rows = evaluate_predictions(label_by_id, predictions_path)
    return SearchResult(
        head_dropout=float(config["head_dropout"]),
        head_hidden_ratio=float(config["head_hidden_ratio"]),
        output_dir=str(predictions_path.parent),
        predictions=str(predictions_path),
        log_loss=loss,
        accuracy=accuracy,
        rows=rows,
        best_trainer_metric=None,
        best_global_step=None,
        status="baseline",
    )


def write_results(path: Path, results: list[SearchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "head_dropout",
        "head_hidden_ratio",
        "log_loss",
        "accuracy",
        "rows",
        "best_trainer_metric",
        "best_global_step",
        "status",
        "error",
        "output_dir",
        "predictions",
    ]
    ordered = sorted(
        results,
        key=lambda item: item.log_loss if item.log_loss is not None else float("inf"),
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
        else search_output_dir / "gemma_mlp_head_hparam_search_results.csv"
    )
    label_by_id = read_labels(Path(args.valid))
    results: list[SearchResult] = []
    baseline = include_baseline(args, label_by_id)
    if baseline is not None:
        results.append(baseline)

    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    for dropout in args.dropouts:
        for ratio in args.hidden_ratios:
            output_dir = search_output_dir / run_name(dropout, ratio)
            predictions_path = output_dir / "gemma2_qlora_valid_predictions.csv"
            command = build_train_command(args, output_dir, dropout, ratio)
            if args.dry_run:
                print(" ".join(command))
                continue
            if args.force or not predictions_path.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                print(f"[search] training head_dropout={dropout} head_hidden_ratio={ratio}")
                try:
                    subprocess.run(command, check=True, env=env, cwd=ROOT)
                except subprocess.CalledProcessError as exc:
                    results.append(
                        SearchResult(
                            head_dropout=dropout,
                            head_hidden_ratio=ratio,
                            output_dir=str(output_dir),
                            predictions=str(predictions_path),
                            log_loss=None,
                            accuracy=None,
                            rows=None,
                            best_trainer_metric=None,
                            best_global_step=None,
                            status="failed",
                            error=f"train.py exited with code {exc.returncode}",
                        )
                    )
                    write_results(results_csv, results)
                    continue
            else:
                print(f"[search] reusing {predictions_path}")

            loss, accuracy, rows = evaluate_predictions(label_by_id, predictions_path)
            best_metric, best_step = trainer_state_metrics(output_dir)
            results.append(
                SearchResult(
                    head_dropout=dropout,
                    head_hidden_ratio=ratio,
                    output_dir=str(output_dir),
                    predictions=str(predictions_path),
                    log_loss=loss,
                    accuracy=accuracy,
                    rows=rows,
                    best_trainer_metric=best_metric,
                    best_global_step=best_step,
                    status="completed",
                )
            )
            write_results(results_csv, results)

    if args.dry_run:
        print("Dry run only.")
        return

    if not results:
        print("No completed results. Dry run only." if args.dry_run else "No results found.")
        return

    write_results(results_csv, results)
    completed = [result for result in results if result.log_loss is not None]
    if not completed:
        print(f"No completed results. Wrote {results_csv}")
        return

    best = min(completed, key=lambda item: item.log_loss)
    print("\nBest MLP head hyperparameters")
    print(f"  head_dropout={best.head_dropout}")
    print(f"  head_hidden_ratio={best.head_hidden_ratio}")
    print(f"  log_loss={best.log_loss:.8f}")
    print(f"  accuracy={best.accuracy:.8f}")
    print(f"  output_dir={best.output_dir}")
    print(f"Wrote {results_csv}")


if __name__ == "__main__":
    main()
