from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from RM_LogisticRegression.models.calibration import (
    average_tta_probabilities,
    augment_with_swapped_responses,
    evaluate_probabilities,
    labels_to_class_ids,
    make_prediction_output,
    swap_response_features,
)
from RM_LogisticRegression.utils.constants import LABEL_COLUMNS, RM_FEATURE_COLUMNS
from RM_LogisticRegression.models.mlp_calibration import set_torch_seed, standardize_features


@dataclass
class PrototypeTrainingResult:
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


class RMPrototypeCalibrator(nn.Module):
    def __init__(
        self,
        initial_prototypes: torch.Tensor,
        learn_feature_weights: bool,
        initial_temperature: float,
    ) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(initial_prototypes.clone())
        self.learn_feature_weights = learn_feature_weights
        self.raw_temperature = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(float(initial_temperature))))
        )
        if learn_feature_weights:
            self.raw_feature_weights = nn.Parameter(
                torch.zeros(initial_prototypes.shape[1], dtype=initial_prototypes.dtype)
            )
        else:
            self.register_buffer(
                "raw_feature_weights",
                torch.zeros(initial_prototypes.shape[1], dtype=initial_prototypes.dtype),
            )

    def feature_weights(self) -> torch.Tensor:
        if self.learn_feature_weights:
            return torch.nn.functional.softplus(self.raw_feature_weights) + 1e-6
        return torch.ones_like(self.raw_feature_weights)

    def temperature(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_temperature) + 1e-6

    def distances(self, features: torch.Tensor) -> torch.Tensor:
        diff = features[:, None, :] - self.prototypes[None, :, :]
        weighted_squared_diff = self.feature_weights()[None, None, :] * diff.square()
        return weighted_squared_diff.sum(dim=2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return -self.distances(features) / self.temperature()


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def class_mean_initial_prototypes(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    prototypes = []
    global_mean = x_train.mean(axis=0)
    for class_id in range(len(LABEL_COLUMNS)):
        class_rows = x_train[y_train == class_id]
        if len(class_rows) == 0:
            prototypes.append(global_mean)
        else:
            prototypes.append(class_rows.mean(axis=0))
    return np.stack(prototypes).astype(np.float32)


def kmeans_plus_plus_initial_prototypes(
    x_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first_index = int(rng.integers(0, len(x_train)))
    selected = [x_train[first_index]]
    min_distances = np.sum((x_train - selected[0]) ** 2, axis=1)

    while len(selected) < len(LABEL_COLUMNS):
        total_distance = float(min_distances.sum())
        if total_distance <= 0.0:
            next_index = int(rng.integers(0, len(x_train)))
        else:
            probabilities = min_distances / total_distance
            next_index = int(rng.choice(len(x_train), p=probabilities))
        selected.append(x_train[next_index])
        candidate_distances = np.sum((x_train - x_train[next_index]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, candidate_distances)

    return np.stack(selected).astype(np.float32)


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
            probability_batches.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(probability_batches, axis=0)


def train_prototype_calibrator(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    init_method: str,
    learn_feature_weights: bool,
    initial_temperature: float,
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
) -> PrototypeTrainingResult:
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
    y_train_soft = train_df[LABEL_COLUMNS].to_numpy(dtype=np.float32)
    y_train_hard = labels_to_class_ids(train_df)
    x_valid_raw = valid_df[RM_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_valid_one_hot = valid_df[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    print(f"  x_train shape: {x_train_raw.shape}")
    print(f"  x_valid shape: {x_valid_raw.shape}")

    x_train, x_valid, feature_mean, feature_std = standardize_features(
        x_train_raw,
        x_valid_raw,
    )

    if init_method == "class-mean":
        initial_prototypes = class_mean_initial_prototypes(x_train, y_train_hard)
    elif init_method == "kmeans++":
        initial_prototypes = kmeans_plus_plus_initial_prototypes(x_train, shuffle_seed)
    else:
        raise ValueError(f"Unknown prototype init method: {init_method}")

    set_torch_seed(shuffle_seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"  device: {device}")
    print(f"  prototype_init: {init_method}")
    print(f"  learn_feature_weights: {learn_feature_weights}")

    model = RMPrototypeCalibrator(
        initial_prototypes=torch.from_numpy(initial_prototypes),
        learn_feature_weights=learn_feature_weights,
        initial_temperature=initial_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    generator = torch.Generator()
    generator.manual_seed(shuffle_seed)
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train_soft),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    x_valid_tensor = torch.from_numpy(x_valid).to(device)
    y_valid_tensor = torch.from_numpy(y_valid_one_hot.astype(np.float32)).to(device)
    best_loss = float("inf")
    best_epoch = 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    print("  fitting supervised prototype calibrator with AdamW")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = soft_cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch_x)
            seen += len(batch_x)

        model.eval()
        with torch.no_grad():
            valid_loss = float(
                soft_cross_entropy(model(x_valid_tensor), y_valid_tensor).item()
            )
        train_loss = train_loss_sum / max(seen, 1)

        if valid_loss < best_loss - min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            temperature = float(model.temperature().detach().cpu().item())
            print(
                f"  epoch={epoch:04d} train_loss={train_loss:.8f} "
                f"valid_loss={valid_loss:.8f} best={best_loss:.8f} "
                f"temperature={temperature:.6f}"
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
        "init_method": init_method,
        "initial_prototypes": initial_prototypes,
        "learn_feature_weights": learn_feature_weights,
        "initial_temperature": initial_temperature,
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
        "final_temperature": float(model.temperature().detach().cpu().item()),
        "final_feature_weights": model.feature_weights().detach().cpu().numpy(),
        "final_prototypes": model.prototypes.detach().cpu().numpy(),
    }
    return PrototypeTrainingResult(
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
