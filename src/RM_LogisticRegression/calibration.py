from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from RM_LogisticRegression.constants import LABEL_COLUMNS, RM_FEATURE_COLUMNS
from RM_LogisticRegression.data import validate_one_hot_labels
from RM_LogisticRegression.metrics import multiclass_accuracy


def read_joined_scores(
    labels_path: Path,
    scores_path: Path,
    allow_partial: bool,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    scores = pd.read_csv(scores_path)
    validate_one_hot_labels(labels, labels_path)

    missing_features = [
        column for column in ["id", *RM_FEATURE_COLUMNS] if column not in scores.columns
    ]
    if missing_features:
        raise ValueError(f"{scores_path} is missing columns: {missing_features}")

    joined = labels[["id", *LABEL_COLUMNS]].merge(
        scores[["id", *RM_FEATURE_COLUMNS]],
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
    return joined


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

    augmented = pd.concat([df, swapped], ignore_index=True)
    return augmented.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_logistic_calibrator(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    c: float,
    max_iter: int,
    augment_swapped: bool,
    shuffle_seed: int,
) -> tuple[pd.DataFrame, float, float, float, object]:
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
    y_valid_one_hot = valid_df[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    y_valid = labels_to_class_ids(valid_df)
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
    probabilities = model.predict_proba(x_valid)
    class_order = list(model.named_steps["logisticregression"].classes_)
    ordered_probabilities = np.zeros((len(valid_df), len(LABEL_COLUMNS)), dtype=np.float64)
    for source_index, class_id in enumerate(class_order):
        ordered_probabilities[:, int(class_id)] = probabilities[:, source_index]

    loss = log_loss(y_valid, ordered_probabilities, labels=[0, 1, 2])
    manual_loss = float(
        -(
            y_valid_one_hot
            * np.log(np.clip(ordered_probabilities, 1e-15, 1.0))
        ).sum(axis=1).mean()
    )
    accuracy = multiclass_accuracy(y_valid_one_hot, ordered_probabilities)

    output = pd.DataFrame({"id": valid_df["id"].to_numpy()})
    for index, column in enumerate(LABEL_COLUMNS):
        output[column] = ordered_probabilities[:, index]
    return output, float(loss), manual_loss, accuracy, model
