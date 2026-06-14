"""
Shared deep encoder blocks and an optional deeper context encoder variant.

The ensemble package mainly reuses ResidualMLPBlock and choose_device from this
file. The deep encoder variant extends a pretrained encoder with projection and
residual MLP comparison features. It is kept for compatibility with the package
and earlier experiments, but the final four-model ensemble uses the context,
pairwise, cross, and single-turn checkpoints.

No Qwen or decoder model is used here.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from EncoderEnsemble.models.blocks import ResidualMLPBlock
from EncoderEnsemble.train_pretrained_preference_model import (
    PreferenceCollator,
    PreferenceTextDataset,
    PreTokenizedPreferenceDataset,
    evaluate_pretrained,
    move_encoded_to_device,
    stack_pretokenized_batch,
)
from EncoderEnsemble.utils.device import choose_device
from EncoderEnsemble.utils.plotting import save_training_plot


class DeepPreferencePretrainedEncoder(nn.Module):
    def __init__(
        self,
        model_name,
        projection_dim=512,
        classifier_layers=3,
        dropout=0.15,
        local_files_only=False,
    ):
        super().__init__()
        # Keep the pretrained backbone unchanged. The extra depth is added after
        # the encoder, so this file follows the current four-input idea.
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden_size = self.backbone.config.hidden_size

        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Features:
        # prompt_a, response_a, prompt_b, response_b,
        # abs(response_a-response_b), response_a*response_b,
        # abs(prompt_a-prompt_b), prompt_a*prompt_b,
        # abs(prompt_a-response_a), prompt_a*response_a,
        # abs(prompt_b-response_b), prompt_b*response_b
        feature_dim = projection_dim * 12
        blocks = [nn.LayerNorm(feature_dim), nn.Linear(feature_dim, projection_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(classifier_layers):
            blocks.append(ResidualMLPBlock(projection_dim, expansion=2, dropout=dropout))
        blocks.extend([nn.LayerNorm(projection_dim), nn.Linear(projection_dim, 3)])
        self.classifier = nn.Sequential(*blocks)

    def encode(self, encoded):
        outputs = self.backbone(**encoded)
        cls = outputs.last_hidden_state[:, 0]
        return self.projection(cls)

    def forward(self, prompt_a_context, prompt_b_context, response_a, response_b, return_logits=False):
        v_pa = self.encode(prompt_a_context)
        v_pb = self.encode(prompt_b_context)
        v_a = self.encode(response_a)
        v_b = self.encode(response_b)

        features = torch.cat(
            [
                v_pa,
                v_a,
                v_pb,
                v_b,
                torch.abs(v_a - v_b),
                v_a * v_b,
                torch.abs(v_pa - v_pb),
                v_pa * v_pb,
                torch.abs(v_pa - v_a),
                v_pa * v_a,
                torch.abs(v_pb - v_b),
                v_pb * v_b,
            ],
            dim=-1,
        )
        logits = self.classifier(features)
        return logits if return_logits else torch.softmax(logits, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default=str(DEFAULT_BASE_PATH))
    parser.add_argument("--processed-train-path", default=None)
    parser.add_argument(
        "--model-name",
        default=local_model_path("deberta-v3-large", "microsoft/deberta-v3-large"),
        help=(
            "Pretrained encoder backbone. Default is a stronger encoder than "
            "distilbert-base-uncased while keeping the same preference-model idea."
        ),
    )
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--valid-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--history-turns", type=int, default=2)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--classifier-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default=output_path("best_deep_pretrained_preference_encoder.pt"))
    parser.add_argument("--plot-path", default=output_path("deep_pretrained_encoder_curves.png"))
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only, use_fast=False)
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

    print("Loading pretrained encoder weights:", args.model_name, flush=True)
    model = DeepPreferencePretrainedEncoder(
        args.model_name,
        projection_dim=args.projection_dim,
        classifier_layers=args.classifier_layers,
        dropout=args.dropout,
        local_files_only=args.local_files_only,
    ).to(device).float()
    if not args.no_gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        print("Gradient checkpointing: enabled")
    print("Deep encoder config:", {
        "projection_dim": args.projection_dim,
        "classifier_layers": args.classifier_layers,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
    })
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
        bar = tqdm(train_loader, desc=f"Deep Encoder Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

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

    if best_epoch == 0:
        print("\nNo training epoch was run, so no validation sample is predicted.")
        return

    sample = valid_df.iloc[0]
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    with torch.no_grad():
        sample_collator = PreferenceCollator(tokenizer, max_length=args.max_length)
        encoded = sample_collator(
            [
                {
                    "prompt_a_context": sample.prompt_a_context,
                    "prompt_b_context": sample.prompt_b_context,
                    "response_a": sample.response_a,
                    "response_b": sample.response_b,
                    "label": int(sample.class_label),
                }
            ]
        )
        probs = model(
            move_encoded_to_device(encoded["prompt_a_context"], device),
            move_encoded_to_device(encoded["prompt_b_context"], device),
            move_encoded_to_device(encoded["response_a"], device),
            move_encoded_to_device(encoded["response_b"], device),
        )[0].cpu()

    pred_label = int(probs.argmax().item())
    true_label = int(sample.class_label)
    print("\nFinal validation sample using best deep encoder checkpoint:")
    print("id:", sample.id)
    print("probabilities:", {name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probs)})
    print("prediction:", CLASS_NAMES[pred_label])
    print("truth:", CLASS_NAMES[true_label])
    print("correct:", pred_label == true_label)


if __name__ == "__main__":
    main()
