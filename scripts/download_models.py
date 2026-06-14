"""
Download missing pretrained model folders for this project.

The encoder ensemble and Gemma2 QLoRA/RM scoring paths expect these
HuggingFace backbones to exist under `models/`:

    models/distilbert-base-uncased
    models/deberta-v3-large
    models/sfairXC__FsfairX-Gemma2-RM-v0.1

This script checks those folders and downloads only the missing or incomplete
ones. It is intentionally limited to pretrained model assets: it does not
download the dataset and does not train checkpoints.

Typical usage:

    python download_models.py

Mirror usage:

    python download_models.py --hf-endpoint https://hf-mirror.com

If the server has no network access, copy the model folders manually into
`models/` and rerun this script to verify that the required files are present.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

from EncoderEnsemble.data.processing import DEFAULT_BASE_PATH


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
MODEL_SPECS = [
    {
        "local_name": "distilbert-base-uncased",
        "repo_id": "distilbert-base-uncased",
    },
    {
        "local_name": "deberta-v3-large",
        "repo_id": "microsoft/deberta-v3-large",
    },
    {
        "local_name": "sfairXC__FsfairX-Gemma2-RM-v0.1",
        "repo_id": "sfairXC/FsfairX-Gemma2-RM-v0.1",
    },
]


def has_any_file(directory, names):
    return any((directory / name).exists() for name in names)


def has_weight_files(model_dir):
    model_dir = Path(model_dir)
    explicit_files = [
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
    ]
    if has_any_file(model_dir, explicit_files):
        return True
    return any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))


def has_tokenizer_files(model_dir):
    model_dir = Path(model_dir)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "spm.model",
        "sentencepiece.bpe.model",
    ]
    return has_any_file(model_dir, tokenizer_files)


def model_ready(model_dir):
    model_dir = Path(model_dir)
    return model_dir.exists() and (model_dir / "config.json").exists() and has_weight_files(model_dir) and has_tokenizer_files(model_dir)


def download_with_hf_cli(repo_id, local_dir, hf_endpoint, dry_run):
    hf_bin = shutil.which("hf")
    if hf_bin is None:
        if dry_run:
            env_text = f"HF_ENDPOINT={hf_endpoint} " if hf_endpoint else ""
            print(f"Would run: {env_text}hf download {repo_id} --local-dir {local_dir}")
            return True
        return False

    cmd = [hf_bin, "download", repo_id, "--local-dir", str(local_dir)]
    env = os.environ.copy()
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    print("Running:", " ".join(cmd))
    if dry_run:
        return True
    subprocess.run(cmd, check=True, env=env)
    return True


def download_with_python_api(repo_id, local_dir, hf_endpoint, dry_run):
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Neither the `hf` command nor the Python package `huggingface_hub` is available.\n"
            "Install one of them first:\n"
            "  pip install -U huggingface_hub"
        ) from exc

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    print(f"Downloading {repo_id} to {local_dir} with huggingface_hub.snapshot_download")
    if dry_run:
        return
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )


def ensure_model(repo_id, local_dir, hf_endpoint, force, dry_run):
    local_dir = Path(local_dir)
    if not force and model_ready(local_dir):
        print(f"Model already ready: {local_dir}")
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model missing or incomplete: {local_dir}")
    used_cli = download_with_hf_cli(repo_id, local_dir, hf_endpoint, dry_run)
    if not used_cli:
        download_with_python_api(repo_id, local_dir, hf_endpoint, dry_run)

    if dry_run:
        print(f"Dry run finished for: {local_dir}")
        return

    if not dry_run and not model_ready(local_dir):
        raise RuntimeError(
            f"Downloaded {repo_id}, but {local_dir} still does not look complete. "
            "Check whether config, weight, and tokenizer files were downloaded."
        )
    print(f"Model ready: {local_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=str(DEFAULT_BASE_PATH))
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--hf-endpoint", default=DEFAULT_HF_ENDPOINT)
    parser.add_argument("--force", action="store_true", help="Redownload even if the local model folder looks complete.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading files.")
    args = parser.parse_args()

    base_path = Path(args.base_path).resolve()
    models_dir = Path(args.models_dir).resolve() if args.models_dir else base_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Base path:", base_path)
    print("Models dir:", models_dir)
    print("HF endpoint:", args.hf_endpoint or "(default HuggingFace)")

    for spec in MODEL_SPECS:
        ensure_model(
            repo_id=spec["repo_id"],
            local_dir=models_dir / spec["local_name"],
            hf_endpoint=args.hf_endpoint,
            force=args.force,
            dry_run=args.dry_run,
        )

    print("Model download check finished.")


if __name__ == "__main__":
    main()
