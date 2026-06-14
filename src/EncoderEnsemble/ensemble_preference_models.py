"""
Validation-time ensemble script for the final four encoder models.

This script loads the context, pairwise, cross, and single-turn checkpoints,
computes their validation probabilities, combines them with fixed ensemble
weights, optionally applies rule-based post-processing, and reports validation
loss, accuracy, and prediction counts.

Only the final four encoder models are included. Qwen candidate models,
decoder models, and tie-detector experiments are intentionally excluded.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from EncoderEnsemble.inference.rule_adjustment import apply_rule_adjustment
from EncoderEnsemble.data.processing import (
    CLASS_NAMES,
    DEFAULT_BASE_PATH,
    load_contextual_data,
    load_cross_data,
    local_model_path,
    output_path,
)
from EncoderEnsemble.train_pretrained_preference_model import (
    PreferenceCollator,
    PreferenceTextDataset,
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


def normalize_weights(weights, active_count):
    weights = weights[:active_count]
    total = sum(weights)
    if total <= 0:
        return [1.0 / active_count] * active_count
    return [w / total for w in weights]


def score_probs(probs, labels):
    loss = F.nll_loss(torch.log(probs.clamp_min(1e-8)), labels).item()
    acc = (probs.argmax(dim=-1) == labels).float().mean().item()
    return loss, acc


def predict_context_model(args, valid_df, device):
    checkpoint = Path(args.context_checkpoint)
    if not checkpoint.exists():
        print(f"Skip context model, checkpoint not found: {checkpoint}")
        return None

    print("Loading context model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.context_model_name, local_files_only=args.local_files_only)
    dataset = PreferenceTextDataset(valid_df)
    loader = DataLoader(
        dataset,
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
        for batch in tqdm(loader, desc="Predict context", dynamic_ncols=True):
            batch_probs = model(
                move_encoded_to_device(batch["prompt_a_context"], device),
                move_encoded_to_device(batch["prompt_b_context"], device),
                move_encoded_to_device(batch["response_a"], device),
                move_encoded_to_device(batch["response_b"], device),
            )
            probs.append(batch_probs.cpu())
    return torch.cat(probs, dim=0)


def predict_pairwise_model(args, valid_df, device):
    checkpoint = Path(args.pairwise_checkpoint)
    if not checkpoint.exists():
        print(f"Skip pairwise model, checkpoint not found: {checkpoint}")
        return None

    print("Loading pairwise model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.pairwise_model_name, local_files_only=args.local_files_only, use_fast=False)
    dataset = PairwiseTextDataset(valid_df)
    loader = DataLoader(
        dataset,
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
        for batch in tqdm(loader, desc="Predict pairwise", dynamic_ncols=True):
            batch_probs = model(
                move_encoded_to_device(batch["pair_a"], device),
                move_encoded_to_device(batch["pair_b"], device),
            )
            probs.append(batch_probs.cpu())
    return torch.cat(probs, dim=0)


def predict_cross_model(args, valid_df, device):
    checkpoint = Path(args.cross_checkpoint)
    if not checkpoint.exists():
        print(f"Skip cross model, checkpoint not found: {checkpoint}")
        return None

    cross_train_df, cross_valid_df = load_cross_data(
        args.base_path,
        args.train_size,
        args.valid_size,
        args.seed,
        args.history_turns,
        processed_train_path=args.processed_train_path,
    )
    import pandas as pd

    cross_by_id = pd.concat([cross_train_df, cross_valid_df], ignore_index=True).set_index("id")
    aligned_rows = []
    missing = 0
    for row in valid_df.itertuples(index=False):
        if row.id in cross_by_id.index:
            aligned_rows.append(cross_by_id.loc[row.id])
        else:
            missing += 1
    if missing:
        raise RuntimeError(f"{missing} ids were not found in cross data.")

    aligned_df = pd.DataFrame(aligned_rows).reset_index(drop=True)

    print("Loading cross model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.cross_model_name, local_files_only=args.local_files_only, use_fast=False)
    dataset = CrossTextDataset(aligned_df)
    loader = DataLoader(
        dataset,
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
        for batch in tqdm(loader, desc="Predict cross", dynamic_ncols=True):
            batch.pop("label")
            batch_probs = model(move_encoded_to_device(batch, device))
            probs.append(batch_probs.cpu())
    return torch.cat(probs, dim=0)


def predict_single_turn_model(args, valid_df, device):
    checkpoint = Path(args.single_turn_checkpoint)
    if not checkpoint.exists():
        print(f"Skip single-turn model, checkpoint not found: {checkpoint}")
        return None

    print("Loading single-turn model:", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.single_turn_model_name, local_files_only=args.local_files_only, use_fast=False)
    single_df = valid_df.copy()
    single_df["sample_weight"] = 1.0
    dataset = SingleTurnTextDataset(single_df)
    loader = DataLoader(
        dataset,
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
        for batch in tqdm(loader, desc="Predict single-turn", dynamic_ncols=True):
            batch.pop("label")
            batch.pop("sample_weight")
            batch_probs = model(move_encoded_to_device(batch, device))
            probs.append(batch_probs.cpu())
    return torch.cat(probs, dim=0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=str(DEFAULT_BASE_PATH))
    parser.add_argument("--processed-train-path", default=None)
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=5000)
    parser.add_argument("--eval-split", choices=["valid", "train"], default="valid")
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

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

    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.1845, 0.3197, 0.3605, 0.1354],
        help="Weights for active models in order: context, pairwise, cross, single_turn.",
    )
    parser.add_argument("--rule-weight", type=float, default=0.2)
    parser.add_argument("--rule-tie-penalty", type=float, default=0.2)
    parser.add_argument("--disable-rule-adjustment", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.allow_cpu)
    train_df, valid_df = load_contextual_data(
        args.base_path,
        args.train_size,
        args.valid_size,
        args.seed,
        args.history_turns,
    )
    eval_df = train_df if args.eval_split == "train" else valid_df
    labels = torch.tensor(eval_df["class_label"].astype(int).tolist(), dtype=torch.long)
    print(f"{args.eval_split.title()} samples: {len(eval_df)}")

    model_outputs = []
    model_names = []

    context_probs = predict_context_model(args, eval_df, device)
    if context_probs is not None:
        model_outputs.append(context_probs)
        model_names.append("context")

    pairwise_probs = predict_pairwise_model(args, eval_df, device)
    if pairwise_probs is not None:
        model_outputs.append(pairwise_probs)
        model_names.append("pairwise")

    cross_probs = predict_cross_model(args, eval_df, device)
    if cross_probs is not None:
        model_outputs.append(cross_probs)
        model_names.append("cross")

    single_turn_probs = predict_single_turn_model(args, eval_df, device)
    if single_turn_probs is not None:
        model_outputs.append(single_turn_probs)
        model_names.append("single_turn")

    if not model_outputs:
        raise RuntimeError("No model checkpoint was loaded. Check checkpoint paths.")

    print(f"\nSingle model {args.eval_split}:")
    for name, probs in zip(model_names, model_outputs):
        loss, acc = score_probs(probs, labels)
        print(f"{name}: valid_loss={loss:.4f} valid_acc={acc:.4f}")

    weights = normalize_weights(args.weights, len(model_outputs))
    ensemble = sum(weight * probs for weight, probs in zip(weights, model_outputs))
    ensemble = ensemble / ensemble.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    loss, acc = score_probs(ensemble, labels)

    print("\nEnsemble:")
    print("models:", model_names)
    print("weights:", [round(w, 4) for w in weights])
    print(f"ensemble_valid_loss={loss:.4f} ensemble_valid_acc={acc:.4f}")

    pred = ensemble.argmax(dim=-1)
    counts = torch.bincount(pred, minlength=3).tolist()
    print("prediction_counts:", {name: count for name, count in zip(CLASS_NAMES, counts)})

    final_probs = ensemble
    final_loss = loss
    final_acc = acc
    final_counts = counts
    final_weights = weights

    rule_report = None
    if not args.disable_rule_adjustment:
        final_probs, rule_report = apply_rule_adjustment(
            final_probs,
            eval_df,
            rule_weight=args.rule_weight,
            tie_penalty=args.rule_tie_penalty,
            return_report=True,
        )
        final_loss, final_acc = score_probs(final_probs, labels)
        final_counts = torch.bincount(final_probs.argmax(dim=-1), minlength=3).tolist()
        print("\nRule adjusted ensemble:")
        print(
            "rule_adjustment:",
            f"matched={rule_report['matched_count']}/{rule_report['total']}",
            f"changed={rule_report['changed_count']}",
            f"weight={args.rule_weight}",
        )
        print(f"rule_adjusted_valid_loss={final_loss:.4f} rule_adjusted_valid_acc={final_acc:.4f}")
        print("rule_adjusted_prediction_counts:", {name: count for name, count in zip(CLASS_NAMES, final_counts)})

    ensemble_config = {
        "models": model_names,
        "weights": [float(w) for w in final_weights],
        "valid_loss": float(final_loss),
        "valid_acc": float(final_acc),
        "prediction_counts": {name: count for name, count in zip(CLASS_NAMES, final_counts)},
        "rule_adjustment": {
            "enabled": not args.disable_rule_adjustment,
            "weight": args.rule_weight,
            "tie_penalty": args.rule_tie_penalty,
            "matched_count": None if rule_report is None else rule_report["matched_count"],
            "changed_count": None if rule_report is None else rule_report["changed_count"],
        },
    }
    if args.eval_split == "valid":
        config_path = output_path("ensemble_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(ensemble_config, f, ensure_ascii=False, indent=2)
        print(f"Saved ensemble config to: {config_path}")
    else:
        print("Skipped saving ensemble_config.json because --eval-split train was used.")


if __name__ == "__main__":
    main()
