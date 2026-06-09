from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from Gemma2_QLoRA.constants import LABEL_COLUMNS


def parse_json_list(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list.")
    return ["" if item is None else str(item) for item in parsed]


def build_compact_pair_text(
    prompt_value: str,
    response_a_value: str,
    response_b_value: str,
) -> str:
    prompts = parse_json_list(prompt_value)
    responses_a = parse_json_list(response_a_value)
    responses_b = parse_json_list(response_b_value)

    turns = []
    for index, prompt in enumerate(prompts):
        response_a = responses_a[index] if index < len(responses_a) else ""
        response_b = responses_b[index] if index < len(responses_b) else ""
        turns.append(
            "<PROMPT>"
            + prompt.strip()
            + "</PROMPT><RESPONSE A>"
            + response_a.strip()
            + "</RESPONSE A><RESPONSE B>"
            + response_b.strip()
            + "</RESPONSE B>"
        )
    return "".join(turns)


def label_id(row: pd.Series) -> int:
    values = row[LABEL_COLUMNS].astype(int).to_numpy()
    if values.sum() != 1:
        raise ValueError(f"Expected one-hot labels for id={row.get('id', '<unknown>')}.")
    return int(values.argmax())


def swap_dataframe(df: pd.DataFrame, swap_labels: bool, suffix_ids: bool) -> pd.DataFrame:
    swapped = df.copy()
    swapped["response_a"], swapped["response_b"] = df["response_b"], df["response_a"]
    if swap_labels:
        swapped["winner_model_a"], swapped["winner_model_b"] = (
            df["winner_model_b"],
            df["winner_model_a"],
        )
    if suffix_ids:
        swapped["id"] = swapped["id"].astype(str) + "_swap"
    return swapped


def add_swap_augmentation(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [df, swap_dataframe(df, swap_labels=True, suffix_ids=True)],
        ignore_index=True,
    )


@dataclass
class PreferenceExample:
    row_id: str
    text: str
    label: int | None
    swapped_text: str | None = None


class PreferenceDataset:
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int,
        limit: int | None = None,
        has_labels: bool = True,
        swap_augmentation: bool = False,
        swap_inputs: bool = False,
        include_swapped_features: bool = False,
    ) -> None:
        df = pd.read_csv(csv_path)
        if limit is not None:
            df = df.head(limit).copy()
        if swap_augmentation:
            if not has_labels:
                raise ValueError("swap_augmentation requires labels.")
            df = add_swap_augmentation(df)
        if swap_inputs:
            df = swap_dataframe(df, swap_labels=False, suffix_ids=False)

        required = ["id", "prompt", "response_a", "response_b"]
        if has_labels:
            required.extend(LABEL_COLUMNS)
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        self.examples = [
            PreferenceExample(
                row_id=str(row["id"]),
                text=build_compact_pair_text(
                    row["prompt"],
                    row["response_a"],
                    row["response_b"],
                ),
                label=label_id(row) if has_labels else None,
                swapped_text=(
                    build_compact_pair_text(
                        row["prompt"],
                        row["response_b"],
                        row["response_a"],
                    )
                    if include_swapped_features
                    else None
                ),
            )
            for _, row in df.iterrows()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        item = dict(encoded)
        item["id"] = example.row_id
        if example.swapped_text is not None:
            swapped_encoded = self.tokenizer(
                example.swapped_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            item["swap_input_ids"] = swapped_encoded["input_ids"]
            item["swap_attention_mask"] = swapped_encoded["attention_mask"]
        if self.has_labels:
            item["labels"] = example.label
        return item


class DataCollatorForPreference:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        ids = [feature.pop("id") for feature in features]
        labels = None
        if "labels" in features[0]:
            labels = torch.tensor([feature.pop("labels") for feature in features])
        swapped_features = None
        if "swap_input_ids" in features[0]:
            swapped_features = [
                {
                    "input_ids": feature.pop("swap_input_ids"),
                    "attention_mask": feature.pop("swap_attention_mask"),
                }
                for feature in features
            ]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        batch["id"] = ids
        if labels is not None:
            batch["labels"] = labels
        if swapped_features is not None:
            swapped_batch = self.tokenizer.pad(
                swapped_features,
                padding=True,
                return_tensors="pt",
            )
            batch["swap_input_ids"] = swapped_batch["input_ids"]
            batch["swap_attention_mask"] = swapped_batch["attention_mask"]
        return batch
