"""
Training script and model definition for the single-turn weighted encoder.

The single-turn encoder focuses on direct current-turn matching between the
prompt and the two candidate responses. During training, later turns can receive
higher sample weights, making the final user request more influential. At
inference time, the model outputs a three-class probability vector for each
sample.

This file can be run directly to train the single-turn checkpoint and is also
imported by the ensemble scripts. No Qwen or decoder model is used here.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from EncoderEnsemble.data.processing import (
    CLASS_NAMES,
    DEFAULT_BASE_PATH,
    format_seconds,
    load_single_turn_data,
    local_model_path,
    output_path,
)
from EncoderEnsemble.utils.device import choose_device
from EncoderEnsemble.train_pretrained_preference_model import move_encoded_to_device
from EncoderEnsemble.utils.plotting import save_training_plot


def build_single_turn_text(prompt, response_a, response_b):
    return (
        "User prompt:\n"
        f"{'' if prompt is None else str(prompt)}\n\n"
        "Response A:\n"
        f"{'' if response_a is None else str(response_a)}\n\n"
        "Response B:\n"
        f"{'' if response_b is None else str(response_b)}"
    )


class SingleTurnTextDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        return {
            "text": build_single_turn_text(row.prompt, row.response_a, row.response_b),
            "label": int(row.class_label),
            "sample_weight": float(row.sample_weight),
        }


class SingleTurnCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        encoded["sample_weight"] = torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float)
        return encoded


class PreTokenizedSingleTurnDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        texts = [
            build_single_turn_text(row.prompt, row.response_a, row.response_b)
            for row in frame.itertuples(index=False)
        ]
        self.encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(frame["class_label"].astype(int).tolist(), dtype=torch.long)
        self.sample_weights = torch.tensor(frame["sample_weight"].astype(float).tolist(), dtype=torch.float)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encoded.items()}
        item["label"] = self.labels[idx]
        item["sample_weight"] = self.sample_weights[idx]
        return item


def stack_single_turn_batch(batch):
    keys = [key for key in batch[0] if key not in {"label", "sample_weight"}]
    result = {key: torch.stack([item[key] for item in batch]) for key in keys}
    result["label"] = torch.stack([item["label"] for item in batch])
    result["sample_weight"] = torch.stack([item["sample_weight"] for item in batch])
    return result


class SingleTurnWeightedEncoder(nn.Module):
    def __init__(self, model_name, classifier_layers=2, dropout=0.15, local_files_only=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_size = self.backbone.config.hidden_size
        layers = [nn.LayerNorm(hidden_size), nn.Dropout(dropout)]
        for _ in range(classifier_layers):
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_size),
                ]
            )
        layers.append(nn.Linear(hidden_size, 3))
        self.classifier = nn.Sequential(*layers)

    def forward(self, encoded, return_logits=False):
        outputs = self.backbone(**encoded)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls)
        return logits if return_logits else torch.softmax(logits, dim=-1)


def split_batch(batch, device):
    labels = batch["label"].to(device, non_blocking=True)
    weights = batch["sample_weight"].to(device, non_blocking=True)
    encoded = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key not in {"label", "sample_weight"}
    }
    return encoded, labels, weights


def weighted_cross_entropy(logits, labels, sample_weights, label_smoothing):
    losses = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
    return (losses * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


def evaluate(model, loader, device, label_smoothing):
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            encoded, labels, sample_weights = split_batch(batch, device)
            logits = model(encoded, return_logits=True)
            loss = weighted_cross_entropy(logits, labels, sample_weights, label_smoothing)
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            seen += batch_size
    return loss_sum / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=str(DEFAULT_BASE_PATH))
    parser.add_argument("--model-name", default=local_model_path("deberta-v3-large", "microsoft/deberta-v3-large"))
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--classifier-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default=output_path("best_single_turn_weighted_encoder.pt"))
    parser.add_argument("--plot-path", default=output_path("single_turn_weighted_curves.png"))
    parser.add_argument("--plot-every", type=int, default=100)
    parser.add_argument("--no-pretokenize", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--hf-endpoint", default=None)
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print("Using HuggingFace endpoint:", args.hf_endpoint)

    torch.manual_seed(args.seed)
    device = choose_device(args.allow_cpu)

    print("Loading weighted single-turn samples...")
    train_df, valid_df = load_single_turn_data(args.base_path, args.train_size, args.valid_size, args.seed)
    print(f"Train expanded turn samples: {len(train_df)} | Valid final-turn samples: {len(valid_df)}")
    print("Train sample weight counts:", train_df["sample_weight"].value_counts().sort_index().to_dict())

    model_config = AutoConfig.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    max_positions = getattr(model_config, "max_position_embeddings", None)
    if max_positions is not None and args.max_length > max_positions:
        print(
            f"Requested --max-length {args.max_length}, but {args.model_name} supports at most "
            f"{max_positions}. Using {max_positions} instead.",
            flush=True,
        )
        args.max_length = max_positions

    print("Loading pretrained tokenizer:", args.model_name, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only, use_fast=False)
    if args.no_pretokenize:
        print("Tokenization mode: on-the-fly per batch")
        train_dataset = SingleTurnTextDataset(train_df)
        valid_dataset = SingleTurnTextDataset(valid_df)
        collate_fn = SingleTurnCollator(tokenizer, args.max_length)
    else:
        print("Tokenization mode: pre-tokenize once and cache tensors")
        print("Pre-tokenizing single-turn prompt + A/B text. This may take a moment.")
        train_dataset = PreTokenizedSingleTurnDataset(train_df, tokenizer, args.max_length)
        valid_dataset = PreTokenizedSingleTurnDataset(valid_df, tokenizer, args.max_length)
        collate_fn = stack_single_turn_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    print("Loading pretrained encoder weights:", args.model_name, flush=True)
    model = SingleTurnWeightedEncoder(
        args.model_name,
        classifier_layers=args.classifier_layers,
        dropout=args.dropout,
        local_files_only=args.local_files_only,
    ).to(device).float()
    if not args.no_gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        print("Gradient checkpointing: enabled")
    print(
        "Single-turn weighted encoder config:",
        {
            "classifier_layers": args.classifier_layers,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "max_length": args.max_length,
            "turn_weights": {"last": 1.0, "previous": 0.5, "earlier": 0.25},
        },
    )
    print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_valid_loss = float("inf")
    best_epoch = 0
    history = []
    progress_history = []
    total_samples = len(train_df) * args.epochs
    total_seen = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        weighted_loss_sum = 0.0
        correct = 0
        seen = 0
        bar = tqdm(train_loader, desc=f"Single Turn Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(bar, start=1):
            encoded, labels, sample_weights = split_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(encoded, return_logits=True)
                loss = weighted_cross_entropy(logits, labels, sample_weights, args.label_smoothing)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            weighted_loss_sum += loss.item() * float(sample_weights.sum().detach().cpu())
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            seen += batch_size
            total_seen += batch_size

            if batch_idx == 1 or batch_idx % max(1, args.plot_every) == 0 or total_seen >= total_samples:
                progress_history.append(
                    {
                        "progress": total_seen / total_samples * 100 if total_samples else 0,
                        "samples": total_seen,
                        "train_loss": loss_sum / seen,
                        "train_acc": correct / seen,
                    }
                )

            elapsed = time.time() - start
            rate = total_seen / elapsed if elapsed > 0 else 0
            eta = (total_samples - total_seen) / rate if rate > 0 else 0
            bar.set_postfix(
                loss=f"{loss_sum / seen:.4f}",
                acc=f"{correct / seen:.4f}",
                samples=f"{seen}/{len(train_df)}",
                total=f"{total_seen}/{total_samples}",
                eta=format_seconds(eta),
            )

        train_loss = loss_sum / seen
        train_acc = correct / seen
        valid_loss, valid_acc = evaluate(model, valid_loader, device, args.label_smoothing)
        history.append(
            {
                "epoch": epoch,
                "progress": total_seen / total_samples * 100 if total_samples else 0,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            }
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.checkpoint_path)

        print(
            f"Epoch {epoch}/{args.epochs} done | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"valid_loss={valid_loss:.4f} valid_acc={valid_acc:.4f} | "
            f"best_epoch={best_epoch} best_valid_loss={best_valid_loss:.4f} | "
            f"elapsed={format_seconds(time.time() - start)}"
        )

    save_training_plot(history, progress_history, args.plot_path)
    if history:
        print(f"Saved training curves to: {args.plot_path}")

    if best_epoch == 0:
        print("\nNo training epoch was run, so no validation sample is predicted.")
        return

    sample = valid_df.iloc[0]
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        sample_collator = SingleTurnCollator(tokenizer, max_length=args.max_length)
        encoded = sample_collator(
            [
                {
                    "text": build_single_turn_text(sample.prompt, sample.response_a, sample.response_b),
                    "label": int(sample.class_label),
                    "sample_weight": 1.0,
                }
            ]
        )
        encoded.pop("sample_weight")
        sample_labels = encoded.pop("label")
        probs = model(move_encoded_to_device(encoded, device))[0].cpu()

    pred_label = int(probs.argmax().item())
    true_label = int(sample_labels[0].item())
    print("\nFinal validation sample using best single-turn weighted checkpoint:")
    print("id:", sample.id)
    print("probabilities:", {name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probs)})
    print("prediction:", CLASS_NAMES[pred_label])
    print("truth:", CLASS_NAMES[true_label])
    print("correct:", pred_label == true_label)


if __name__ == "__main__":
    main()
