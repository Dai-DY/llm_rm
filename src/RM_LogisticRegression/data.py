import json
from pathlib import Path

import pandas as pd

from RM_LogisticRegression.constants import LABEL_COLUMNS


def parse_json_list(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")
    return ["" if item is None else str(item) for item in parsed]


def text_char_length(value: str) -> int:
    return sum(len(item) for item in parse_json_list(value))


def build_reward_text(prompt_value: str, response_value: str) -> str:
    prompts = parse_json_list(prompt_value)
    responses = parse_json_list(response_value)
    parts = []
    for index, prompt in enumerate(prompts):
        parts.append(f"User: {prompt.strip()}")
        if index < len(responses):
            parts.append(f"Assistant: {responses[index].strip()}")
    return "\n\n".join(parts)


def validate_one_hot_labels(df: pd.DataFrame, path: Path | None = None) -> None:
    missing = [column for column in LABEL_COLUMNS if column not in df.columns]
    if missing:
        location = f"{path} " if path is not None else ""
        raise ValueError(f"{location}is missing label columns: {missing}")

    row_sums = df[LABEL_COLUMNS].sum(axis=1)
    if not (row_sums == 1).all():
        bad_count = int((row_sums != 1).sum())
        raise ValueError(f"Expected one-hot labels; found {bad_count} invalid rows.")


def label_names(df: pd.DataFrame) -> pd.Series:
    validate_one_hot_labels(df)
    return df[LABEL_COLUMNS].idxmax(axis=1)


def print_label_distribution(name: str, df: pd.DataFrame) -> None:
    counts = df[LABEL_COLUMNS].sum()
    rates = counts / len(df)
    joined = ", ".join(
        f"{column}={int(counts[column])} ({rates[column]:.4f})"
        for column in LABEL_COLUMNS
    )
    print(f"{name}: rows={len(df)}, {joined}")


def read_required_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df

