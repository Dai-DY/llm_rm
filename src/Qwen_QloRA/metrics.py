import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def multiclass_log_loss(
    labels: np.ndarray,
    probabilities: np.ndarray,
    clip: float = 1e-15,
) -> float:
    probabilities = np.clip(probabilities, clip, 1.0 - clip)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return float(-np.log(probabilities[np.arange(len(labels)), labels]).mean())


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    probabilities = softmax(logits)
    return {"log_loss": multiclass_log_loss(labels, probabilities)}

