"""
Training script and model definition for the context encoder.

The context encoder separately encodes prompt_a_context, response_a,
prompt_b_context, and response_b with a pretrained encoder backbone. The four
vectors are concatenated and passed to a three-class classifier predicting
A win, B win, or Tie.

This file can be run directly to train the context checkpoint, and its dataset,
collator, model class, and helper functions are reused by the ensemble scripts.
No Qwen or decoder model is used here.
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
    ensure_context_columns,
    format_seconds,
    load_contextual_data,
    load_data,
    local_model_path,
    output_path,
)
from EncoderEnsemble.utils.plotting import save_training_plot


class PreferenceTextDataset(Dataset):
    """Raw-text dataset; tokenization is done batch-wise in collate_fn."""

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
            "label": int(row.class_label),
        }


class PreTokenizedPreferenceDataset(Dataset):
    # CHANGE: Speed optimization.
    # Tokenize prompt/response_a/response_b once before training instead of
    # repeating tokenization inside every epoch and every batch.
    def __init__(self, frame, tokenizer, max_length):
        self.labels = torch.tensor(frame["class_label"].astype(int).tolist(), dtype=torch.long)
        self.prompt_a_context = self.tokenize_column(frame["prompt_a_context"], tokenizer, max_length)
        self.prompt_b_context = self.tokenize_column(frame["prompt_b_context"], tokenizer, max_length)
        self.response_a = self.tokenize_column(frame["response_a"], tokenizer, max_length)
        self.response_b = self.tokenize_column(frame["response_b"], tokenizer, max_length)

    @staticmethod
    def tokenize_column(series, tokenizer, max_length):
        texts = series.fillna("").astype(str).tolist()
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "prompt_a_context": {key: value[idx] for key, value in self.prompt_a_context.items()},
            "prompt_b_context": {key: value[idx] for key, value in self.prompt_b_context.items()},
            "response_a": {key: value[idx] for key, value in self.response_a.items()},
            "response_b": {key: value[idx] for key, value in self.response_b.items()},
            "label": self.labels[idx],
        }


def stack_pretokenized_batch(batch):
    def stack_encoded(name):
        return {
            key: torch.stack([item[name][key] for item in batch])
            for key in batch[0][name]
        }

    return {
        "prompt_a_context": stack_encoded("prompt_a_context"),
        "prompt_b_context": stack_encoded("prompt_b_context"),
        "response_a": stack_encoded("response_a"),
        "response_b": stack_encoded("response_b"),
        "label": torch.stack([item["label"] for item in batch]),
    }


class PreferenceCollator:
    # CHANGE: Use a pretrained tokenizer instead of the simple regex tokenizer.
    # The tokenizer knows subwords, special tokens, padding, truncation, and attention masks.
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_texts(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, batch):
        return {
            "prompt_a_context": self.tokenize_texts([item["prompt_a_context"] for item in batch]),
            "prompt_b_context": self.tokenize_texts([item["prompt_b_context"] for item in batch]),
            "response_a": self.tokenize_texts([item["response_a"] for item in batch]),
            "response_b": self.tokenize_texts([item["response_b"] for item in batch]),
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        }


class PreferencePretrainedModel(nn.Module):
    def __init__(self, model_name, dropout=0.1, local_files_only=False):
        super().__init__()
        # CHANGE: Replace the from-scratch embedding + SimpleTransformerBlock stack
        # with a pretrained language-model backbone from HuggingFace.
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_size = self.backbone.config.hidden_size

        # CHANGE: Use two prompt-context branches:
        # prompt_a_context contains current prompt plus A's recent history.
        # prompt_b_context contains current prompt plus B's recent history.
        # response_a/response_b still contain only the current answer.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 4, 3)

    def encode(self, encoded):
        outputs = self.backbone(**encoded)
        # CHANGE: Use the pretrained model's CLS/first-token vector as sentence representation.
        return outputs.last_hidden_state[:, 0]

    def forward(self, prompt_a_context, prompt_b_context, response_a, response_b, return_logits=False):
        v_pa = self.encode(prompt_a_context)
        v_pb = self.encode(prompt_b_context)
        v_a = self.encode(response_a)
        v_b = self.encode(response_b)
        logits = self.classifier(self.dropout(torch.cat([v_pa, v_a, v_pb, v_b], dim=-1)))
        return logits if return_logits else torch.softmax(logits, dim=-1)


def move_encoded_to_device(encoded, device):
    return {key: value.to(device, non_blocking=True) for key, value in encoded.items()}


def evaluate_pretrained(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for batch in loader:
            prompt_a_context = move_encoded_to_device(batch["prompt_a_context"], device)
            prompt_b_context = move_encoded_to_device(batch["prompt_b_context"], device)
            response_a = move_encoded_to_device(batch["response_a"], device)
            response_b = move_encoded_to_device(batch["response_b"], device)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(prompt_a_context, prompt_b_context, response_a, response_b, return_logits=True)
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

    # CHANGE: Defaults are smaller than the custom Transformer scripts because
    # pretrained backbones are heavier and each batch runs the backbone three times.
    parser.add_argument("--model-name", default=local_model_path("distilbert-base-uncased"))
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--valid-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=2,
        help=(
            "How many previous turns to pack into each AI-specific prompt context. "
            "Use 0 for only the current prompt."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.02,
        help="Label smoothing for CrossEntropyLoss. The original notebook used 0.02.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-path",
        default=output_path("best_context_pretrained_preference_model.pt"),
        help="Where to save the best checkpoint for this context-aware four-input model.",
    )
    parser.add_argument(
        "--plot-path",
        default=output_path("pretrained_training_curves.png"),
        help="Where to save the loss/accuracy visualization after training.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=100,
        help="Record one training-curve point every N batches.",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--no-pretokenize",
        action="store_true",
        help="Disable pre-tokenization cache. Slower, but uses less RAM.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only cached HuggingFace files. Helpful on offline machines.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional HuggingFace mirror endpoint, e.g. https://hf-mirror.com",
    )
    args = parser.parse_args()
   
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print("Using HuggingFace endpoint:", args.hf_endpoint)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    elif args.allow_cpu:
        device = torch.device("cpu")
        print("Using CPU because --allow-cpu was set.")
    else:
        raise RuntimeError("CUDA GPU is not available. Use a CUDA kernel or pass --allow-cpu for debugging.")

    if args.history_turns > 0:
        print(f"Loading raw multi-turn train samples with history_turns={args.history_turns}...")
        train_df, valid_df = load_contextual_data(
            args.base_path,
            args.train_size,
            args.valid_size,
            args.seed,
            args.history_turns,
        )
    else:
        print("Loading notebook-processed train samples...")
        train_df, valid_df = load_data(
            args.base_path,
            args.train_size,
            args.valid_size,
            args.seed,
            processed_train_path=args.processed_train_path,
        )
        train_df = ensure_context_columns(train_df)
        valid_df = ensure_context_columns(valid_df)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    if args.no_pretokenize:
        print("Tokenization mode: on-the-fly per batch")
        collator = PreferenceCollator(tokenizer, max_length=args.max_length)
        train_dataset = PreferenceTextDataset(train_df)
        valid_dataset = PreferenceTextDataset(valid_df)
        collate_fn = collator
    else:
        print("Tokenization mode: pre-tokenize once and cache tensors")
        print("Pre-tokenizing train/valid text. This may take a moment, but training epochs will be faster.")
        train_dataset = PreTokenizedPreferenceDataset(train_df, tokenizer, args.max_length)
        valid_dataset = PreTokenizedPreferenceDataset(valid_df, tokenizer, args.max_length)
        collate_fn = stack_pretokenized_batch

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

    print("Loading pretrained model weights:", args.model_name, flush=True)
    print("If this line appears to hang, the model weights are being downloaded or the network is blocked.", flush=True)
    model = PreferencePretrainedModel(
        args.model_name,
        dropout=args.dropout,
        local_files_only=args.local_files_only,
    ).to(device)
    model.float()
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
        bar = tqdm(train_loader, desc=f"Pretrained Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for batch_idx, batch in enumerate(bar, start=1):
            prompt_a_context = move_encoded_to_device(batch["prompt_a_context"], device)
            prompt_b_context = move_encoded_to_device(batch["prompt_b_context"], device)
            response_a = move_encoded_to_device(batch["response_a"], device)
            response_b = move_encoded_to_device(batch["response_b"], device)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(prompt_a_context, prompt_b_context, response_a, response_b, return_logits=True)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            seen += batch_size
            total_seen += batch_size

            should_record_progress = batch_idx == 1 or batch_idx % max(1, args.plot_every) == 0 or total_seen >= total_samples
            if should_record_progress:
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
        valid_loss, valid_acc = evaluate_pretrained(model, valid_loader, criterion, device)
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

    sample = valid_df.iloc[0]
    if best_epoch == 0:
        print("\nNo training epoch was run, so no validation sample is predicted.")
        return
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        sample_collator = PreferenceCollator(tokenizer, max_length=args.max_length)
        encoded = sample_collator([{
            "prompt_a_context": sample.prompt_a_context,
            "prompt_b_context": sample.prompt_b_context,
            "response_a": sample.response_a,
            "response_b": sample.response_b,
            "label": int(sample.class_label),
        }])
        probs = model(
            move_encoded_to_device(encoded["prompt_a_context"], device),
            move_encoded_to_device(encoded["prompt_b_context"], device),
            move_encoded_to_device(encoded["response_a"], device),
            move_encoded_to_device(encoded["response_b"], device),
        )[0].cpu()

    pred_label = int(probs.argmax().item())
    true_label = int(sample.class_label)
    print("\nFinal validation sample using best pretrained checkpoint:")
    print("id:", sample.id)
    print("probabilities:", {name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probs)})
    print("prediction:", CLASS_NAMES[pred_label])
    print("truth:", CLASS_NAMES[true_label])
    print("correct:", pred_label == true_label)


if __name__ == "__main__":
    main()
