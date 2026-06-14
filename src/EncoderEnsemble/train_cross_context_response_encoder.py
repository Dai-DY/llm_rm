"""
Training script and model definition for the cross context-response encoder.

The cross encoder places shared_context, response_a, and response_b into one
sequence so that self-attention can compare the two candidate responses inside
a single encoder pass. The first-token representation is passed to a three-class
classifier.

This file can be run directly to train the cross checkpoint and is also imported
by the ensemble scripts. No Qwen or decoder model is used here.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from EncoderEnsemble.data.processing import (
    CLASS_NAMES,
    DEFAULT_BASE_PATH,
    format_seconds,
    load_cross_data,
    local_model_path,
    output_path,
)
from EncoderEnsemble.train_pretrained_preference_model import move_encoded_to_device
from EncoderEnsemble.utils.device import choose_device
from EncoderEnsemble.utils.plotting import save_training_plot


def build_cross_text(shared_context, response_a, response_b):
    # CHANGE: One encoder input contains the shared prompt context and both
    # candidate answers, so self-attention can compare A and B directly.
    return (
        "Shared conversation context:\n"
        f"{'' if shared_context is None else str(shared_context)}\n\n"
        "Response A:\n"
        f"{'' if response_a is None else str(response_a)}\n\n"
        "Response B:\n"
        f"{'' if response_b is None else str(response_b)}"
    )


class CrossTextDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        return {
            "text": build_cross_text(row.shared_context, row.response_a, row.response_b),
            "label": int(row.class_label),
        }


class CrossCollator:
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
        return encoded


class PreTokenizedCrossDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        texts = [
            build_cross_text(row.shared_context, row.response_a, row.response_b)
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

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encoded.items()}
        item["label"] = self.labels[idx]
        return item


def stack_cross_batch(batch):
    keys = [key for key in batch[0] if key != "label"]
    result = {key: torch.stack([item[key] for item in batch]) for key in keys}
    result["label"] = torch.stack([item["label"] for item in batch])
    return result


class CrossContextResponseEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        classifier_layers=2,
        dropout=0.15,
        local_files_only=False,
    ):
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


def split_encoded_and_labels(batch, device):
    labels = batch["label"].to(device, non_blocking=True)
    encoded = {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "label"
    }
    return encoded, labels


def evaluate_cross(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            encoded, labels = split_encoded_and_labels(batch, device)
            logits = model(encoded, return_logits=True)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            seen += batch_size
    return loss_sum / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=str(DEFAULT_BASE_PATH))
    parser.add_argument("--processed-train-path", default=None)
    parser.add_argument("--model-name", default=local_model_path("deberta-v3-large", "microsoft/deberta-v3-large"))
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--classifier-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default=output_path("best_cross_context_response_encoder.pt"))
    parser.add_argument("--plot-path", default=output_path("cross_context_response_curves.png"))
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

    print(f"Loading shared-context train samples with history_turns={args.history_turns}...")
    train_df, valid_df = load_cross_data(
        args.base_path,
        args.train_size,
        args.valid_size,
        args.seed,
        args.history_turns,
        processed_train_path=args.processed_train_path,
    )
    print(f"Train samples: {len(train_df)} | Valid samples: {len(valid_df)}")

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
        train_dataset = CrossTextDataset(train_df)
        valid_dataset = CrossTextDataset(valid_df)
        collate_fn = CrossCollator(tokenizer, args.max_length)
    else:
        print("Tokenization mode: pre-tokenize once and cache tensors")
        print("Pre-tokenizing shared context + A/B text. This may take a moment.")
        train_dataset = PreTokenizedCrossDataset(train_df, tokenizer, args.max_length)
        valid_dataset = PreTokenizedCrossDataset(valid_df, tokenizer, args.max_length)
        collate_fn = stack_cross_batch

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
    model = CrossContextResponseEncoder(
        args.model_name,
        classifier_layers=args.classifier_layers,
        dropout=args.dropout,
        local_files_only=args.local_files_only,
    ).to(device).float()
    if not args.no_gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        print("Gradient checkpointing: enabled")
    print(
        "Cross encoder config:",
        {
            "classifier_layers": args.classifier_layers,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "max_length": args.max_length,
        },
    )
    print("Trainable parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
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
        correct = 0
        seen = 0
        bar = tqdm(train_loader, desc=f"Cross Encoder Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(bar, start=1):
            encoded, labels = split_encoded_and_labels(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(encoded, return_logits=True)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
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
        valid_loss, valid_acc = evaluate_cross(model, valid_loader, criterion, device)
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
        sample_collator = CrossCollator(tokenizer, max_length=args.max_length)
        encoded = sample_collator(
            [
                {
                    "text": build_cross_text(sample.shared_context, sample.response_a, sample.response_b),
                    "label": int(sample.class_label),
                }
            ]
        )
        sample_labels = encoded.pop("label")
        probs = model(move_encoded_to_device(encoded, device))[0].cpu()

    pred_label = int(probs.argmax().item())
    true_label = int(sample_labels[0].item())
    print("\nFinal validation sample using best cross checkpoint:")
    print("id:", sample.id)
    print("probabilities:", {name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probs)})
    print("prediction:", CLASS_NAMES[pred_label])
    print("truth:", CLASS_NAMES[true_label])
    print("correct:", pred_label == true_label)


if __name__ == "__main__":
    main()
