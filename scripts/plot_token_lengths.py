import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token length distribution from a token-length CSV."
    )
    parser.add_argument(
        "--input",
        default="output/train_split_gemma2_token_lengths.csv",
        help="CSV containing id and token_length columns.",
    )
    parser.add_argument(
        "--output",
        default="output/train_split_gemma2_token_lengths.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1800,
        help="Reference max length to draw and count truncation risk.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of longest examples to show.",
    )
    return parser.parse_args()


def read_lengths(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"id": str})
    missing = [column for column in ["id", "token_length"] if column not in data.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    data = data[["id", "token_length"]].copy()
    data["token_length"] = pd.to_numeric(data["token_length"], errors="raise")
    return data


def add_reference_lines(axis, max_length: int, ymax: float | None = None) -> None:
    axis.axvline(
        max_length,
        color="#d1495b",
        linewidth=1.8,
        linestyle="--",
        label=f"max_length={max_length}",
    )
    if ymax is not None:
        axis.set_ylim(top=ymax)


def plot_token_lengths(
    data: pd.DataFrame,
    output_path: Path,
    max_length: int,
    bins: int,
    top_k: int,
) -> None:
    lengths = data["token_length"]
    p50 = lengths.quantile(0.50)
    p95 = lengths.quantile(0.95)

    fig, hist_axis = plt.subplots(figsize=(11, 6), constrained_layout=True)

    hist_axis.hist(
        lengths,
        bins=bins,
        color="#3f7cac",
        edgecolor="white",
        linewidth=0.4,
        alpha=0.88,
    )
    hist_axis.set_title("Train Token Length Distribution", fontsize=16, fontweight="bold")
    hist_axis.set_xlabel("Token length")
    hist_axis.set_ylabel("Rows")
    hist_axis.grid(axis="y", alpha=0.25)
    hist_axis.text(
        0.98,
        0.92,
        (
            f"rows: {len(data):,}\n"
            f"mean: {lengths.mean():.1f}\n"
            f"p50: {p50:.0f}\n"
            f"p95: {p95:.0f}"
        ),
        transform=hist_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#c8c8c8"},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    data = read_lengths(input_path)
    plot_token_lengths(
        data=data,
        output_path=output_path,
        max_length=args.max_length,
        bins=args.bins,
        top_k=args.top_k,
    )
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
