"""
Data processing utilities for the four-model preference prediction ensemble.

This module reads the raw LMSYS/Kaggle-style train and test CSV files, parses
multi-turn prompt and response lists, extracts the current turn, builds recent
conversation context fields, and creates the labels used by the classifier
models. It is shared by the training scripts, validation ensemble script, and
test-time submission script.

Expected data layout under the project root:
    data/train.csv
    data/test.csv

The module does not train models and does not depend on any Qwen or decoder
components.
"""

import ast
import json
from pathlib import Path

import pandas as pd


DEFAULT_BASE_PATH = Path(__file__).resolve().parents[3]
CLASS_NAMES = ["A胜", "B胜", "平局"]
LABEL_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]
NAME_TO_LABEL = {"winner_model_a": 0, "winner_model_b": 1, "winner_tie": 2}
NOTEBOOK_PROCESSED_COLUMNS = ["id", "prompt", "response_a", "response_b", "class_name", "class_label"]


def local_model_path(model_dir_name, fallback_name=None):
    path = DEFAULT_BASE_PATH / "models" / model_dir_name
    if path.exists():
        return str(path)
    return fallback_name or model_dir_name


def output_path(filename):
    output_dir = DEFAULT_BASE_PATH / "output" / "EncoderEnsemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / filename)


def split_csv_path(base_path, split):
    base_path = Path(base_path)
    candidates = [
        base_path / "data" / f"{split}.csv",
        base_path / split / f"{split}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_text_list(value):
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        parsed = ast.literal_eval(str(value).replace("null", "None"))
    if not isinstance(parsed, list):
        return []
    return ["" if item is None else str(item) for item in parsed]


def parse_first(value):
    items = parse_text_list(value)
    return items[0] if items else ""


def format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"


def choose_current_turn(prompts, responses_a, responses_b):
    if not prompts:
        return 0
    current_idx = min(len(prompts), len(responses_a), len(responses_b)) - 1
    return max(0, current_idx)


def build_ai_prompt_context(prompts, responses, current_idx, history_turns):
    start = max(0, current_idx - history_turns)
    parts = []
    for turn_idx in range(start, current_idx):
        if turn_idx < len(prompts):
            parts.append(f"Previous user: {prompts[turn_idx]}")
        if turn_idx < len(responses):
            parts.append(f"Previous assistant: {responses[turn_idx]}")
    if current_idx < len(prompts):
        parts.append(f"Current user: {prompts[current_idx]}")
    return "\n".join(parts)


def build_shared_context(prompts, responses_a, responses_b, current_idx, history_turns):
    start = max(0, current_idx - history_turns)
    parts = []
    for turn_idx in range(start, current_idx):
        if turn_idx < len(prompts):
            parts.append(f"Previous user: {prompts[turn_idx]}")
        if turn_idx < len(responses_a):
            parts.append(f"Previous response A: {responses_a[turn_idx]}")
        if turn_idx < len(responses_b):
            parts.append(f"Previous response B: {responses_b[turn_idx]}")
    if current_idx < len(prompts):
        parts.append(f"Current user: {prompts[current_idx]}")
    return "\n".join(parts)


def build_contextual_frame(base_path, history_turns, split="train"):
    raw_df = pd.read_csv(split_csv_path(base_path, split))
    rows = []
    has_labels = all(name in raw_df.columns for name in LABEL_COLUMNS)

    for row in raw_df.itertuples(index=False):
        prompts = parse_text_list(row.prompt)
        responses_a = parse_text_list(row.response_a)
        responses_b = parse_text_list(row.response_b)
        current_idx = choose_current_turn(prompts, responses_a, responses_b)

        item = {
            "id": row.id,
            "prompt": prompts[current_idx] if current_idx < len(prompts) else "",
            "prompt_a_context": build_ai_prompt_context(prompts, responses_a, current_idx, history_turns),
            "prompt_b_context": build_ai_prompt_context(prompts, responses_b, current_idx, history_turns),
            "shared_context": build_shared_context(prompts, responses_a, responses_b, current_idx, history_turns),
            "response_a": responses_a[current_idx] if current_idx < len(responses_a) else "",
            "response_b": responses_b[current_idx] if current_idx < len(responses_b) else "",
        }
        if has_labels:
            class_name = max(LABEL_COLUMNS, key=lambda name: getattr(row, name))
            item["class_name"] = class_name
            item["class_label"] = NAME_TO_LABEL[class_name]
        rows.append(item)

    return pd.DataFrame(rows)


def turn_sample_weight(turn_idx, last_idx):
    distance = last_idx - turn_idx
    if distance <= 0:
        return 1.0
    if distance == 1:
        return 0.5
    return 0.25


def build_single_turn_frame(base_path, split="train", include_all_turns=True):
    raw_df = pd.read_csv(split_csv_path(base_path, split))
    rows = []
    has_labels = all(name in raw_df.columns for name in LABEL_COLUMNS)

    for row in raw_df.itertuples(index=False):
        prompts = parse_text_list(row.prompt)
        responses_a = parse_text_list(row.response_a)
        responses_b = parse_text_list(row.response_b)
        if not prompts:
            prompts = [""]
        last_idx = choose_current_turn(prompts, responses_a, responses_b)
        turn_indices = range(last_idx + 1) if include_all_turns else [last_idx]

        class_name = None
        class_label = None
        if has_labels:
            class_name = max(LABEL_COLUMNS, key=lambda name: getattr(row, name))
            class_label = NAME_TO_LABEL[class_name]

        for turn_idx in turn_indices:
            item = {
                "id": row.id,
                "turn_idx": turn_idx,
                "last_turn_idx": last_idx,
                "prompt": prompts[turn_idx] if turn_idx < len(prompts) else "",
                "response_a": responses_a[turn_idx] if turn_idx < len(responses_a) else "",
                "response_b": responses_b[turn_idx] if turn_idx < len(responses_b) else "",
                "sample_weight": turn_sample_weight(turn_idx, last_idx),
            }
            if has_labels:
                item["class_name"] = class_name
                item["class_label"] = class_label
            rows.append(item)

    return pd.DataFrame(rows)


def load_contextual_data(base_path, train_size, valid_size, seed, history_turns):
    df = build_contextual_frame(base_path, history_turns, split="train")
    n = min(train_size + valid_size, len(df))
    work_df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return work_df.iloc[:train_size].copy(), work_df.iloc[train_size:].copy()


def load_single_turn_data(base_path, train_size, valid_size, seed):
    base_df = build_single_turn_frame(base_path, split="train", include_all_turns=False)
    n = min(train_size + valid_size, len(base_df))
    sampled_ids = base_df.sample(n=n, random_state=seed)["id"].tolist()
    train_ids = set(sampled_ids[:train_size])
    valid_ids = set(sampled_ids[train_size:])

    all_turns_df = build_single_turn_frame(base_path, split="train", include_all_turns=True)
    train_df = all_turns_df[all_turns_df["id"].isin(train_ids)].copy().reset_index(drop=True)
    valid_df = base_df[base_df["id"].isin(valid_ids)].copy().reset_index(drop=True)
    return train_df, valid_df


def build_notebook_processed_frame(base_path):
    raw_df = pd.read_csv(split_csv_path(base_path, "train"))
    df = raw_df.copy()
    df["prompt"] = df["prompt"].map(parse_first)
    df["response_a"] = df["response_a"].map(parse_first)
    df["response_b"] = df["response_b"].map(parse_first)
    df["class_name"] = df[LABEL_COLUMNS].idxmax(axis=1)
    df["class_label"] = df["class_name"].map(NAME_TO_LABEL)
    return df


def load_notebook_processed_frame(base_path, processed_train_path=None):
    if processed_train_path:
        processed_path = Path(processed_train_path)
    else:
        processed_path = Path(base_path) / "notebook_train_processed.csv"

    if processed_path.exists():
        print(f"Loading notebook processed data: {processed_path}")
        df = pd.read_csv(processed_path)
    else:
        print("Processed notebook data file not found.")
        print("Recreating notebook-processed df from data/train.csv with the same parsing logic.")
        df = build_notebook_processed_frame(base_path)

    missing = [col for col in NOTEBOOK_PROCESSED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Processed data is missing required columns: {missing}. "
            "Expected data exported from the notebook after prompt/response parsing."
        )

    df = df.copy()
    df["prompt"] = df["prompt"].fillna("")
    df["response_a"] = df["response_a"].fillna("")
    df["response_b"] = df["response_b"].fillna("")
    df["class_label"] = df["class_label"].astype(int)
    return df


def load_data(base_path, train_size, valid_size, seed, processed_train_path=None):
    df = load_notebook_processed_frame(base_path, processed_train_path)
    n = min(train_size + valid_size, len(df))
    work_df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return work_df.iloc[:train_size].copy(), work_df.iloc[train_size:].copy()


def load_cross_data(base_path, train_size, valid_size, seed, history_turns, processed_train_path=None):
    del processed_train_path
    return load_contextual_data(base_path, train_size, valid_size, seed, history_turns)


def ensure_context_columns(frame):
    frame = frame.copy()
    if "prompt_a_context" not in frame.columns:
        frame["prompt_a_context"] = frame["prompt"]
    if "prompt_b_context" not in frame.columns:
        frame["prompt_b_context"] = frame["prompt"]
    if "shared_context" not in frame.columns:
        frame["shared_context"] = frame["prompt"]
    return frame
