import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Gemma2_QLoRA.constants import LABEL_COLUMNS
from Gemma2_QLoRA.metrics import multiclass_log_loss, softmax
from Gemma2_QLoRA.modeling import MLPClassificationHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only the Gemma2 MLP classification head from saved pooled features."
    )
    parser.add_argument("--train-features", required=True, help="Train .pt from extract script.")
    parser.add_argument("--valid-features", required=True, help="Validation .pt from extract script.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint and outputs.")
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--head-hidden-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help='Use "auto", "cuda", "cpu", or a specific device such as "cuda:0".',
    )
    parser.add_argument(
        "--predictions-name",
        default="gemma2_mlp_head_valid_predictions.csv",
        help="Validation prediction CSV filename.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="gemma2_mlp_head.pt",
        help="Head checkpoint filename.",
    )
    return parser.parse_args()


def torch_load(path: str | Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def labels_to_numpy(labels: torch.Tensor | None, path: str | Path) -> np.ndarray:
    if labels is None:
        raise ValueError(f"{path} does not contain labels. Re-run extraction with --has-labels.")
    return labels.detach().cpu().numpy().astype(np.int64)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_probabilities(
    model: nn.Module,
    features: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size].to(device)
            logits = model(batch).detach().float().cpu().numpy()
            batches.append(softmax(logits))
    return np.concatenate(batches, axis=0)


def write_predictions(
    ids: list[str],
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    rows = []
    for row_id, probs in zip(ids, probabilities):
        rows.append(
            {
                "id": row_id,
                LABEL_COLUMNS[0]: probs[0],
                LABEL_COLUMNS[1]: probs[1],
                LABEL_COLUMNS[2]: probs[2],
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / args.checkpoint_name
    predictions_path = output_dir / args.predictions_name
    metrics_path = output_dir / "gemma2_mlp_head_metrics.json"

    train_payload = torch_load(args.train_features)
    valid_payload = torch_load(args.valid_features)
    x_train = train_payload["features"].detach().float()
    y_train = torch.from_numpy(labels_to_numpy(train_payload["labels"], args.train_features))
    x_valid = valid_payload["features"].detach().float()
    y_valid_np = labels_to_numpy(valid_payload["labels"], args.valid_features)
    valid_ids = [str(row_id) for row_id in valid_payload["ids"]]

    if x_train.ndim != 2 or x_valid.ndim != 2:
        raise ValueError("Expected feature tensors with shape [rows, hidden_size].")
    if x_train.size(1) != x_valid.size(1):
        raise ValueError(
            f"Train/valid hidden sizes differ: {x_train.size(1)} vs {x_valid.size(1)}"
        )

    device = resolve_device(args.device)
    print(f"device: {device}")
    print(f"train features: {tuple(x_train.shape)}")
    print(f"valid features: {tuple(x_valid.shape)}")
    print(f"head_dropout: {args.head_dropout}")
    print(f"head_hidden_ratio: {args.head_hidden_ratio}")

    model = MLPClassificationHead(
        hidden_size=x_train.size(1),
        num_labels=len(LABEL_COLUMNS),
        dropout=args.head_dropout,
        hidden_ratio=args.head_hidden_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch_x)
            seen += len(batch_x)

        probabilities = predict_probabilities(model, x_valid, device, args.batch_size)
        valid_loss = multiclass_log_loss(y_valid_np, probabilities)
        train_loss = train_loss_sum / max(seen, 1)
        print(
            f"epoch {epoch:03d} "
            f"train_loss={train_loss:.6f} valid_log_loss={valid_loss:.6f}"
        )

        if valid_loss < best_loss - args.min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    probabilities = predict_probabilities(model, x_valid, device, args.batch_size)
    predictions_loss = multiclass_log_loss(y_valid_np, probabilities)
    accuracy = float((probabilities.argmax(axis=1) == y_valid_np).mean())

    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "hidden_size": x_train.size(1),
            "num_labels": len(LABEL_COLUMNS),
            "head_dropout": args.head_dropout,
            "head_hidden_ratio": args.head_hidden_ratio,
            "label_columns": LABEL_COLUMNS,
        },
        "train_features_metadata": train_payload.get("metadata", {}),
        "valid_features_metadata": valid_payload.get("metadata", {}),
        "metrics": {
            "best_epoch": best_epoch,
            "valid_log_loss": predictions_loss,
            "valid_accuracy": accuracy,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    write_predictions(valid_ids, probabilities, predictions_path)
    metrics_path.write_text(
        json.dumps(checkpoint["metrics"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"best_epoch: {best_epoch}")
    print(f"valid_log_loss: {predictions_loss:.6f}")
    print(f"valid_accuracy: {accuracy:.6f}")
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved predictions: {predictions_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
