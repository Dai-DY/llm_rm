import argparse
import shutil
from pathlib import Path

import _bootstrap  # noqa: F401


DEFAULT_OUTPUT_SPECS = {
    "gemma2": {
        "dataset": "daidysh643/gemma-finetune-output",
        "output_dir": "output/gemma2_finetune",
    },
    "encoder_ensemble": {
        "dataset": "zxhddyl/encoderensemble",
        "output_dir": "output/EncoderEnsemble",
    },
}


def copy_downloaded_files(source_dir: Path, output_dir: Path, dry_run: bool) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"KaggleHub returned a path that does not exist: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        target = output_dir / item.name
        if dry_run:
            print(f"Would copy {item} -> {target}")
            continue
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        print(f"Copied {item} -> {target}")


def output_exists(output_dir: Path) -> bool:
    return output_dir.exists() and any(output_dir.iterdir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download trained model outputs from KaggleHub into output/."
    )
    parser.add_argument(
        "--target",
        choices=["all", *DEFAULT_OUTPUT_SPECS.keys()],
        default="all",
        help="Which trained output dataset to download.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override KaggleHub dataset handle. Only valid when --target is not all.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override destination directory. Only valid when --target is not all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download/check the dataset path and print copy actions without writing files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and copy even when the destination directory already has files.",
    )
    return parser.parse_args()


def selected_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.target == "all":
        if args.dataset is not None or args.output_dir is not None:
            raise ValueError("--dataset and --output-dir overrides require --target to be a single dataset.")
        return list(DEFAULT_OUTPUT_SPECS.values())

    spec = DEFAULT_OUTPUT_SPECS[args.target].copy()
    if args.dataset is not None:
        spec["dataset"] = args.dataset
    if args.output_dir is not None:
        spec["output_dir"] = args.output_dir
    return [spec]


def main() -> None:
    args = parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Could not import kagglehub. Install or repair it first:\n"
            "  pip install -U kagglehub kagglesdk"
        ) from exc

    for spec in selected_specs(args):
        output_dir = Path(spec["output_dir"]).resolve()
        print(f"Dataset: {spec['dataset']}")
        print(f"Output dir: {output_dir}")
        if output_exists(output_dir) and not args.force:
            print(f"Output already exists, skip download: {output_dir}")
            continue

        downloaded_path = Path(kagglehub.dataset_download(spec["dataset"])).resolve()
        print(f"Path to dataset files: {downloaded_path}")
        copy_downloaded_files(downloaded_path, output_dir, args.dry_run)
    print("Trained output download finished.")


if __name__ == "__main__":
    main()
