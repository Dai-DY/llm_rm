import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Gemma2_QLoRA.constants import LABEL_COLUMNS
from Gemma2_QLoRA.metrics import softmax
from Gemma2_QLoRA.modeling import MLPClassificationHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict probabilities with a saved Gemma2 MLP head and pooled features."
    )
    parser.add_argument("--features", required=True, help=".pt feature file from extraction.")
    parser.add_argument("--checkpoint", required=True, help="Head checkpoint from training.")
    parser.add_argument("--output", required=True, help="Output probability CSV.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        default="auto",
        help='Use "auto", "cuda", "cpu", or a specific device such as "cuda:0".',
    )
    return parser.parse_args()


def torch_load(path: str | Path):
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def predict_probabilities(
    model: torch.nn.Module,
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


def main() -> None:
    args = parse_args()
    feature_payload = torch_load(args.features)
    checkpoint = torch_load(args.checkpoint)
    config = checkpoint["config"]

    features = feature_payload["features"].detach().float()
    ids = [str(row_id) for row_id in feature_payload["ids"]]
    device = resolve_device(args.device)

    model = MLPClassificationHead(
        hidden_size=int(config["hidden_size"]),
        num_labels=int(config["num_labels"]),
        dropout=float(config["head_dropout"]),
        hidden_ratio=float(config["head_hidden_ratio"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    probabilities = predict_probabilities(model, features, device, args.batch_size)
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"features: {tuple(features.shape)}")
    print(f"saved predictions: {output_path}")


if __name__ == "__main__":
    main()
