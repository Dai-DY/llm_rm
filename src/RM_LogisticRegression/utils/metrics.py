import numpy as np
import pandas as pd

from RM_LogisticRegression.utils.constants import LABEL_COLUMNS


def prepare_prediction_probabilities(
    predictions: pd.DataFrame,
    normalize: bool,
) -> np.ndarray:
    probs = predictions[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError("Predictions contain NaN or infinite values.")
    if (probs < 0).any():
        raise ValueError("Predictions contain negative probabilities.")

    row_sums = probs.sum(axis=1, keepdims=True)
    if normalize:
        if (row_sums <= 0).any():
            raise ValueError("Cannot normalize rows with non-positive sums.")
        probs = probs / row_sums
    elif not np.allclose(row_sums, 1.0, atol=1e-4):
        min_sum = float(row_sums.min())
        max_sum = float(row_sums.max())
        raise ValueError(
            "Prediction rows must sum to 1. "
            f"Observed min={min_sum:.6f}, max={max_sum:.6f}. "
            "Use --normalize to normalize them before evaluation."
        )

    return probs


def multiclass_log_loss(y_true: np.ndarray, y_pred: np.ndarray, clip: float) -> float:
    y_pred = np.clip(y_pred, clip, 1.0 - clip)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return float(-(y_true * np.log(y_pred)).sum(axis=1).mean())


def multiclass_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_labels = y_true.argmax(axis=1)
    pred_labels = y_pred.argmax(axis=1)
    return float((true_labels == pred_labels).mean())
