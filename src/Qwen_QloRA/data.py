from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from Qwen_QloRA.constants import LABEL_COLUMNS


def parse_json_list(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")
    return ["" if item is None else str(item) for item in parsed]


def format_list_block(title: str, values: list[str]) -> str:
    lines = [f"{title}:"]
    for index, value in enumerate(values, start=1):
        lines.append(f"[{index}] {value.strip()}")
    return "\n".join(lines)


def build_pair_text(prompt_value: str, response_a_value: str, response_b_value: str) -> str:
    prompts = parse_json_list(prompt_value)
    responses_a = parse_json_list(response_a_value)
    responses_b = parse_json_list(response_b_value)

    return "\n\n".join(
        [
            "Task: Predict which assistant response is preferred by human voters.",
            format_list_block("Prompt turns", prompts),
            format_list_block("Response A turns", responses_a),
            format_list_block("Response B turns", responses_b),
            "Classes: winner_model_a, winner_model_b, winner_tie.",
        ]
    )


def label_id(row: pd.Series) -> int:
    values = row[LABEL_COLUMNS].astype(int).to_numpy()
    if values.sum() != 1:
        raise ValueError(f"Expected one-hot labels for id={row.get('id', '<unknown>')}.")
    return int(values.argmax())


def add_swap_augmentation(df: pd.DataFrame) -> pd.DataFrame:
    swapped = df.copy()
    swapped["response_a"], swapped["response_b"] = df["response_b"], df["response_a"]
    swapped["winner_model_a"], swapped["winner_model_b"] = (
        df["winner_model_b"],
        df["winner_model_a"],
    )
    swapped["id"] = swapped["id"].astype(str) + "_swap"
    return pd.concat([df, swapped], ignore_index=True)


@dataclass
class PreferenceExample:
    row_id: str
    text: str
    label: int | None


class PreferenceDataset:
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int,
        limit: int | None = None,
        has_labels: bool = True,
        swap_augmentation: bool = False,
    ) -> None:
        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.head(limit).copy()
        if swap_augmentation:
            if not has_labels:
                raise ValueError("swap_augmentation requires labels.")
            df = add_swap_augmentation(df)

        required = ["id", "prompt", "response_a", "response_b"]
        if has_labels:
            required.extend(LABEL_COLUMNS)
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        self.examples = [
            PreferenceExample(
                row_id=str(row["id"]),
                text=build_pair_text(row["prompt"], row["response_a"], row["response_b"]),
                label=label_id(row) if has_labels else None,
            )
            for _, row in df.iterrows()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item: dict[str, list[int] | int | str] = dict(encoded)
        item["id"] = example.row_id
        if self.has_labels:
            item["labels"] = example.label
        return item


class DataCollatorForPreference:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        import torch

        ids = [feature.pop("id") for feature in features]
        labels = None
        if "labels" in features[0]:
            labels = torch.tensor([feature.pop("labels") for feature in features])

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        batch["id"] = ids
        if labels is not None:
            batch["labels"] = labels
        return batch
