from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from RM_LogisticRegression.utils.constants import (
    LABEL_COLUMNS,
    RM_BASE_FEATURE_COLUMNS,
    RM_FEATURE_COLUMNS,
)
from RM_LogisticRegression.data.processing import validate_one_hot_labels
from RM_LogisticRegression.utils.metrics import multiclass_accuracy


@dataclass
class CalibrationResult:
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
    model: object


def read_joined_scores(
    labels_path: Path,
    scores_path: Path,
    allow_partial: bool,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    scores = pd.read_csv(scores_path)
    validate_one_hot_labels(labels, labels_path)

    missing_features = [
        column
        for column in ["id", *RM_BASE_FEATURE_COLUMNS]
        if column not in scores.columns
    ]
    if missing_features:
        raise ValueError(f"{scores_path} is missing columns: {missing_features}")

    joined = labels[["id", *LABEL_COLUMNS]].merge(
        scores[["id", *RM_BASE_FEATURE_COLUMNS]],
        on="id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(labels) and not allow_partial:
        raise ValueError(
            f"{scores_path} matched {len(joined)} of {len(labels)} label rows."
        )
    if len(joined) == 0:
        raise ValueError(f"{scores_path} did not match any label rows.")
    return add_derived_features(joined)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["score_sum"] = df["score_a"] + df["score_b"]
    df["score_mean"] = df["score_sum"] / 2.0
    df["score_product"] = df["score_a"] * df["score_b"]
    df["score_max"] = df[["score_a", "score_b"]].max(axis=1)
    df["score_min"] = df[["score_a", "score_b"]].min(axis=1)

    df["response_len_abs_diff"] = df["response_len_diff"].abs()
    df["response_len_sum"] = df["response_a_len"] + df["response_b_len"]
    df["response_len_mean"] = df["response_len_sum"] / 2.0
    df["response_len_ratio"] = (
        (df["response_a_len"] + 1.0) / (df["response_b_len"] + 1.0)
    )
    df["log_prompt_len"] = np.log1p(df["prompt_len"].clip(lower=0))
    df["log_response_a_len"] = np.log1p(df["response_a_len"].clip(lower=0))
    df["log_response_b_len"] = np.log1p(df["response_b_len"].clip(lower=0))
    return df


def labels_to_class_ids(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL_COLUMNS].to_numpy().argmax(axis=1)


def augment_with_swapped_responses(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    swapped = df.copy()
    swapped["id"] = swapped["id"].astype(str) + "_swap"

    swapped["winner_model_a"] = df["winner_model_b"].to_numpy()
    swapped["winner_model_b"] = df["winner_model_a"].to_numpy()
    swapped["winner_tie"] = df["winner_tie"].to_numpy()

    swapped["score_a"] = df["score_b"].to_numpy()
    swapped["score_b"] = df["score_a"].to_numpy()
    swapped["score_diff"] = -df["score_diff"].to_numpy()
    swapped["score_abs_diff"] = df["score_abs_diff"].to_numpy()
    swapped["response_a_len"] = df["response_b_len"].to_numpy()
    swapped["response_b_len"] = df["response_a_len"].to_numpy()
    swapped["response_len_diff"] = -df["response_len_diff"].to_numpy()
    swapped["prompt_len"] = df["prompt_len"].to_numpy()
    swapped = add_derived_features(swapped)

    augmented = pd.concat([df, swapped], ignore_index=True)
    return augmented.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def swap_response_features(df: pd.DataFrame) -> pd.DataFrame:
    swapped = df.copy()
    swapped["score_a"] = df["score_b"].to_numpy()
    swapped["score_b"] = df["score_a"].to_numpy()
    swapped["score_diff"] = -df["score_diff"].to_numpy()
    swapped["score_abs_diff"] = df["score_abs_diff"].to_numpy()
    swapped["response_a_len"] = df["response_b_len"].to_numpy()
    swapped["response_b_len"] = df["response_a_len"].to_numpy()
    swapped["response_len_diff"] = -df["response_len_diff"].to_numpy()
    swapped["prompt_len"] = df["prompt_len"].to_numpy()
    return add_derived_features(swapped)


def map_swapped_probabilities_to_original(probabilities: np.ndarray) -> np.ndarray:
    mapped = probabilities.copy()
    mapped[:, 0] = probabilities[:, 1]
    mapped[:, 1] = probabilities[:, 0]
    mapped[:, 2] = probabilities[:, 2]
    return mapped


def average_tta_probabilities(
    probabilities: np.ndarray,
    swapped_probabilities: np.ndarray,
) -> np.ndarray:
    averaged = (
        probabilities + map_swapped_probabilities_to_original(swapped_probabilities)
    ) / 2.0
    return averaged / averaged.sum(axis=1, keepdims=True)


def make_prediction_output(
    valid_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    output = pd.DataFrame({"id": valid_df["id"].to_numpy()})
    for index, column in enumerate(LABEL_COLUMNS):
        output[column] = probabilities[:, index]
    return output


def evaluate_probabilities(
    valid_df: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[float, float, float]:
    y_valid_one_hot = valid_df[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    y_valid = labels_to_class_ids(valid_df)
    loss = log_loss(y_valid, probabilities, labels=[0, 1, 2])
    manual_loss = float(
        -(
            y_valid_one_hot
            * np.log(np.clip(probabilities, 1e-15, 1.0))
        ).sum(axis=1).mean()
    )
    accuracy = multiclass_accuracy(y_valid_one_hot, probabilities)
    return float(loss), manual_loss, accuracy


def ordered_predict_proba(model: object, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(features)
    class_order = list(model.named_steps["logisticregression"].classes_)
    ordered_probabilities = np.zeros(
        (len(features), len(LABEL_COLUMNS)),
        dtype=np.float64,
    )
    for source_index, class_id in enumerate(class_order):
        ordered_probabilities[:, int(class_id)] = probabilities[:, source_index]
    return ordered_probabilities


def train_logistic_calibrator(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    c: float,
    max_iter: int,
    augment_swapped: bool,
    shuffle_seed: int,
    valid_tta: bool = False,
) -> CalibrationResult:
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

    x_train = train_df[RM_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_train = labels_to_class_ids(train_df)
    x_valid = valid_df[RM_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    print(f"  x_train shape: {x_train.shape}")
    print(f"  x_valid shape: {x_valid.shape}")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c,
            max_iter=max_iter,
            solver="lbfgs",
        ),
    )
    print("  fitting StandardScaler + LogisticRegression")
    model.fit(x_train, y_train)

    print("  predicting validation probabilities")
    base_probabilities = ordered_predict_proba(model, x_valid)
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
        x_valid_swapped = swapped_valid_df[RM_FEATURE_COLUMNS].to_numpy(
            dtype=np.float64,
        )
        swapped_probabilities = ordered_predict_proba(model, x_valid_swapped)
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

    return CalibrationResult(
        output=output,
        loss=loss,
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
        model=model,
    )
