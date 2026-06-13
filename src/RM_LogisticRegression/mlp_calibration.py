from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from RM_LogisticRegression.calibration import (
    average_tta_probabilities,
    augment_with_swapped_responses,
    evaluate_probabilities,
    labels_to_class_ids,
    make_prediction_output,
    swap_response_features,
)
from RM_LogisticRegression.constants import LABEL_COLUMNS, RM_FEATURE_COLUMNS


@dataclass
class MLPTrainingResult:
    output: pd.DataFrame
    loss: float
    manual_loss: float
    accuracy: float
    base_output: pd.DataFrame
    base_loss: float
    base_manual_loss: float
    base_accuracy: float
    tta_output: pd.DataFrame | None
    tta_loss: float | None
    tta_manual_loss: float | None
    tta_accuracy: float | None
    checkpoint: dict


class RMCalibratorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, len(LABEL_COLUMNS))),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, len(LABEL_COLUMNS)), len(LABEL_COLUMNS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize_features(
    x_train: np.ndarray,
    x_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_valid - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def predict_probabilities(
    model: nn.Module,
    x_valid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probability_batches = []
    with torch.no_grad():
        for start in range(0, len(x_valid), batch_size):
            batch = torch.from_numpy(x_valid[start : start + batch_size]).to(device)
            probability_batches.append(
                torch.softmax(model(batch), dim=1).cpu().numpy()
            )
    return np.concatenate(probability_batches, axis=0)


def train_mlp_calibrator(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    augment_swapped: bool,
    shuffle_seed: int,
    device_name: str,
    valid_tta: bool = False,
) -> MLPTrainingResult:
    print("  preparing feature matrices")
    if augment_swapped:
        original_rows = len(train_df)
        train_df = augment_with_swapped_responses(train_df, shuffle_seed)
        print(
            "  augmented train rows with swapped A/B responses: "
            f"{original_rows} -> {len(train_df)}"
        )
        print(f"  shuffled augmented train rows with seed={shuffle_seed}")
    else:
        print("  swapped-response augmentation disabled")

    x_train_raw = train_df[RM_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = labels_to_class_ids(train_df).astype(np.int64)
    x_valid_raw = valid_df[RM_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_valid = labels_to_class_ids(valid_df)
    print(f"  x_train shape: {x_train_raw.shape}")
    print(f"  x_valid shape: {x_valid_raw.shape}")

    x_train, x_valid, feature_mean, feature_std = standardize_features(
        x_train_raw,
        x_valid_raw,
    )

    set_torch_seed(shuffle_seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"  device: {device}")

    model = RMCalibratorMLP(
        input_dim=len(RM_FEATURE_COLUMNS),
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator()
    generator.manual_seed(shuffle_seed)
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    x_valid_tensor = torch.from_numpy(x_valid).to(device)
    y_valid_tensor = torch.from_numpy(y_valid.astype(np.int64)).to(device)
    best_loss = float("inf")
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    print("  fitting StandardScaler + MLP with AdamW")
    for epoch in range(1, max_epochs + 1):
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

        model.eval()
        with torch.no_grad():
            valid_loss = float(criterion(model(x_valid_tensor), y_valid_tensor).item())
        train_loss = train_loss_sum / max(seen, 1)

        if valid_loss < best_loss - min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(
                f"  epoch={epoch:04d} train_loss={train_loss:.8f} "
                f"valid_loss={valid_loss:.8f} best={best_loss:.8f}"
            )

        if epoch - best_epoch >= patience:
            print(
                "  early stopping: "
                f"best_epoch={best_epoch}, best_valid_loss={best_loss:.8f}"
            )
            break

    model.load_state_dict(best_state)
    print("  predicting validation probabilities")
    base_probabilities = predict_probabilities(model, x_valid, device, batch_size)

    base_loss, base_manual_loss, base_accuracy = evaluate_probabilities(
        valid_df,
        base_probabilities,
    )
    base_output = make_prediction_output(valid_df, base_probabilities)

    tta_output = None
    tta_loss = None
    tta_manual_loss = None
    tta_accuracy = None
    output = base_output
    loss = base_loss
    manual_loss = base_manual_loss
    accuracy = base_accuracy
    if valid_tta:
        print("  predicting swapped validation probabilities for TTA")
        swapped_valid_df = swap_response_features(valid_df)
        x_valid_swapped_raw = swapped_valid_df[RM_FEATURE_COLUMNS].to_numpy(
            dtype=np.float32,
        )
        x_valid_swapped = ((x_valid_swapped_raw - feature_mean) / feature_std).astype(
            np.float32,
        )
        swapped_probabilities = predict_probabilities(
            model,
            x_valid_swapped,
            device,
            batch_size,
        )
        tta_probabilities = average_tta_probabilities(
            base_probabilities,
            swapped_probabilities,
        )
        tta_loss, tta_manual_loss, tta_accuracy = evaluate_probabilities(
            valid_df,
            tta_probabilities,
        )
        tta_output = make_prediction_output(valid_df, tta_probabilities)
        output = tta_output
        loss = tta_loss
        manual_loss = tta_manual_loss
        accuracy = tta_accuracy

    checkpoint = {
        "model_state_dict": best_state,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_columns": RM_FEATURE_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "valid_tta": valid_tta,
        "validation_base_log_loss": base_loss,
        "validation_base_accuracy": base_accuracy,
        "validation_tta_log_loss": tta_loss,
        "validation_tta_accuracy": tta_accuracy,
    }
    return MLPTrainingResult(
        output=output,
        loss=float(loss),
        manual_loss=manual_loss,
        accuracy=accuracy,
        base_output=base_output,
        base_loss=base_loss,
        base_manual_loss=base_manual_loss,
        base_accuracy=base_accuracy,
        tta_output=tta_output,
        tta_loss=tta_loss,
        tta_manual_loss=tta_manual_loss,
        tta_accuracy=tta_accuracy,
        checkpoint=checkpoint,
    )
