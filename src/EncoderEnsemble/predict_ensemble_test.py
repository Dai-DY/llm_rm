"""
Test-time prediction script for the final four-model encoder ensemble.

This script reads data/test.csv, builds the same context fields used during
training, loads the four encoder checkpoints, averages their probability outputs
with the configured ensemble weights, optionally applies rule-based
post-processing, and writes a Kaggle-style submission.csv file.

Only the final four encoder models are included. Qwen candidate models,
decoder models, and tie-detector experiments are intentionally excluded.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from EncoderEnsemble.ensemble_preference_models import normalize_weights
from EncoderEnsemble.inference.rule_adjustment import apply_rule_adjustment
from EncoderEnsemble.data.processing import (
    DEFAULT_BASE_PATH,
    LABEL_COLUMNS,
    build_contextual_frame,
    local_model_path,
    output_path,
)
from EncoderEnsemble.train_pretrained_preference_model import (
    PreferenceCollator,
    PreferencePretrainedModel,
    move_encoded_to_device,
)
from EncoderEnsemble.train_pairwise_context_response_encoder import (
    PairwiseCollator,
    PairwiseContextResponseEncoder,
    PairwiseTextDataset,
)
from EncoderEnsemble.train_cross_context_response_encoder import (
    CrossCollator,
    CrossContextResponseEncoder,
    CrossTextDataset,
)
from EncoderEnsemble.train_single_turn_weighted_encoder import (
    SingleTurnCollator,
    SingleTurnTextDataset,
    SingleTurnWeightedEncoder,
)
from EncoderEnsemble.utils.device import choose_device


def build_test_frame(base_path, history_turns):
    frame = build_contextual_frame(base_path, history_turns, split="test")
    frame["class_label"] = 0
    return frame


def load_ensemble_config(path):
    config_path = Path(path)
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def predict_context(args, test_df, device):
    checkpoint = Path(args.context_checkpoint)
    if not checkpoint.exists():
        print(f"Skip context model, checkpoint not found: {checkpoint}")
        return None

    print("Predict test with context model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.context_model_name, local_files_only=args.local_files_only)
    loader = DataLoader(
        ContextTextDataset(test_df),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PreferenceCollator(tokenizer, args.max_length),
        pin_memory=device.type == "cuda",
    )
    model = PreferencePretrainedModel(
        args.context_model_name,
        dropout=args.context_dropout,
        local_files_only=args.local_files_only,
    ).to(device).float()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test context", dynamic_ncols=True):
            probs.append(
                model(
                    move_encoded_to_device(batch["prompt_a_context"], device),
                    move_encoded_to_device(batch["prompt_b_context"], device),
                    move_encoded_to_device(batch["response_a"], device),
                    move_encoded_to_device(batch["response_b"], device),
                ).cpu()
            )
    return torch.cat(probs, dim=0)


class ContextTextDataset:
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        return {
            "prompt_a_context": "" if row.prompt_a_context is None else str(row.prompt_a_context),
            "prompt_b_context": "" if row.prompt_b_context is None else str(row.prompt_b_context),
            "response_a": "" if row.response_a is None else str(row.response_a),
            "response_b": "" if row.response_b is None else str(row.response_b),
            "label": 0,
        }


def predict_pairwise(args, test_df, device):
    checkpoint = Path(args.pairwise_checkpoint)
    if not checkpoint.exists():
        print(f"Skip pairwise model, checkpoint not found: {checkpoint}")
        return None

    print("Predict test with pairwise model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.pairwise_model_name, local_files_only=args.local_files_only, use_fast=False)
    loader = DataLoader(
        PairwiseTextDataset(test_df),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PairwiseCollator(tokenizer, args.max_length),
        pin_memory=device.type == "cuda",
    )
    model = PairwiseContextResponseEncoder(
        args.pairwise_model_name,
        projection_dim=args.projection_dim,
        classifier_layers=args.classifier_layers,
        dropout=args.pairwise_dropout,
        local_files_only=args.local_files_only,
    ).to(device).float()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test pairwise", dynamic_ncols=True):
            probs.append(
                model(
                    move_encoded_to_device(batch["pair_a"], device),
                    move_encoded_to_device(batch["pair_b"], device),
                ).cpu()
            )
    return torch.cat(probs, dim=0)


def predict_cross(args, test_df, device):
    checkpoint = Path(args.cross_checkpoint)
    if not checkpoint.exists():
        print(f"Skip cross model, checkpoint not found: {checkpoint}")
        return None

    print("Predict test with cross model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.cross_model_name, local_files_only=args.local_files_only, use_fast=False)
    loader = DataLoader(
        CrossTextDataset(test_df),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CrossCollator(tokenizer, args.max_length),
        pin_memory=device.type == "cuda",
    )
    model = CrossContextResponseEncoder(
        args.cross_model_name,
        classifier_layers=args.cross_classifier_layers,
        dropout=args.cross_dropout,
        local_files_only=args.local_files_only,
    ).to(device).float()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test cross", dynamic_ncols=True):
            batch.pop("label")
            probs.append(model(move_encoded_to_device(batch, device)).cpu())
    return torch.cat(probs, dim=0)


def predict_single_turn(args, test_df, device):
    checkpoint = Path(args.single_turn_checkpoint)
    if not checkpoint.exists():
        print(f"Skip single-turn model, checkpoint not found: {checkpoint}")
        return None

    print("Predict test with single-turn model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.single_turn_model_name, local_files_only=args.local_files_only, use_fast=False)
    single_df = test_df.copy()
    single_df["sample_weight"] = 1.0
    loader = DataLoader(
        SingleTurnTextDataset(single_df),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=SingleTurnCollator(tokenizer, args.max_length),
        pin_memory=device.type == "cuda",
    )
    model = SingleTurnWeightedEncoder(
        args.single_turn_model_name,
        classifier_layers=args.single_turn_classifier_layers,
        dropout=args.single_turn_dropout,
        local_files_only=args.local_files_only,
    ).to(device).float()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test single-turn", dynamic_ncols=True):
            batch.pop("label")
            batch.pop("sample_weight")
            probs.append(model(move_encoded_to_device(batch, device)).cpu())
    return torch.cat(probs, dim=0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=str(DEFAULT_BASE_PATH))
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-path", default=output_path("submission.csv"))
    parser.add_argument("--ensemble-config", default=output_path("ensemble_config.json"))

    parser.add_argument("--context-checkpoint", default=output_path("best_context_pretrained_preference_model.pt"))
    parser.add_argument("--context-model-name", default=local_model_path("distilbert-base-uncased"))
    parser.add_argument("--context-dropout", type=float, default=0.1)

    parser.add_argument("--pairwise-checkpoint", default=output_path("best_pairwise_context_response_encoder.pt"))
    parser.add_argument("--pairwise-model-name", default=local_model_path("deberta-v3-large", "microsoft/deberta-v3-large"))
    parser.add_argument("--pairwise-dropout", type=float, default=0.15)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--classifier-layers", type=int, default=3)

    parser.add_argument("--cross-checkpoint", default=output_path("best_cross_context_response_encoder.pt"))
    parser.add_argument("--cross-model-name", default=local_model_path("deberta-v3-large", "microsoft/deberta-v3-large"))
    parser.add_argument("--cross-dropout", type=float, default=0.15)
    parser.add_argument("--cross-classifier-layers", type=int, default=2)

    parser.add_argument("--single-turn-checkpoint", default=output_path("best_single_turn_weighted_encoder.pt"))
    parser.add_argument("--single-turn-model-name", default=local_model_path("deberta-v3-large", "microsoft/deberta-v3-large"))
    parser.add_argument("--single-turn-dropout", type=float, default=0.15)
    parser.add_argument("--single-turn-classifier-layers", type=int, default=2)


    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--rule-weight", type=float, default=0.8)
    parser.add_argument("--rule-tie-penalty", type=float, default=0.2)
    parser.add_argument("--disable-rule-adjustment", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.allow_cpu)
    test_df = build_test_frame(args.base_path, args.history_turns)
    print(f"Test samples: {len(test_df)}")

    model_outputs = []
    model_names = []

    for name, predict_fn in [
        ("context", predict_context),
        ("pairwise", predict_pairwise),
        ("cross", predict_cross),
        ("single_turn", predict_single_turn),
    ]:
        probs = predict_fn(args, test_df, device)
        if probs is not None:
            model_names.append(name)
            model_outputs.append(probs)

    if not model_outputs:
        raise RuntimeError("No model checkpoint was loaded. Check checkpoint paths.")

    config = load_ensemble_config(args.ensemble_config)
    if args.weights is not None:
        weights = normalize_weights(args.weights, len(model_outputs))
    elif config and config.get("models") == model_names:
        weights = normalize_weights(config["weights"], len(model_outputs))
    else:
        if config:
            print(f"Ignore ensemble config because model order differs: {config.get('models')} != {model_names}")
        weights = [1.0 / len(model_outputs)] * len(model_outputs)

    print("Ensemble models:", model_names)
    print("Ensemble weights:", [round(w, 4) for w in weights])

    ensemble = sum(weight * probs for weight, probs in zip(weights, model_outputs))
    ensemble = ensemble / ensemble.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    if not args.disable_rule_adjustment:
        ensemble, rule_report = apply_rule_adjustment(
            ensemble,
            test_df,
            rule_weight=args.rule_weight,
            tie_penalty=args.rule_tie_penalty,
            return_report=True,
        )
        print(
            "Rule adjustment:",
            f"matched={rule_report['matched_count']}/{rule_report['total']}",
            f"changed={rule_report['changed_count']}",
            f"weight={args.rule_weight}",
        )

    submission = pd.DataFrame({"id": test_df["id"]})
    for idx, column in enumerate(LABEL_COLUMNS):
        submission[column] = ensemble[:, idx].numpy()
    submission.to_csv(args.output_path, index=False)
    print(f"Saved submission to: {args.output_path}")
    print("Prediction counts:", submission[LABEL_COLUMNS].idxmax(axis=1).value_counts().to_dict())


if __name__ == "__main__":
    main()
