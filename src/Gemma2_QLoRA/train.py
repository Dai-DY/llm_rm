import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Gemma2_QLoRA.constants import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, LABEL_COLUMNS
from Gemma2_QLoRA.data import DataCollatorForPreference, PreferenceDataset
from Gemma2_QLoRA.metrics import multiclass_log_loss, softmax
from Gemma2_QLoRA.modeling import (
    load_gemma2_sequence_classifier,
    load_tokenizer,
    parse_target_modules,
)


def make_training_arguments(training_arguments_cls, **kwargs):
    import inspect

    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return training_arguments_cls(**kwargs)


def trainer_compute_metrics(eval_pred) -> dict[str, float]:
    logits = getattr(eval_pred, "predictions", eval_pred[0])
    labels = getattr(eval_pred, "label_ids", eval_pred[1])
    return {"log_loss": multiclass_log_loss(labels, softmax(logits))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma2 RM with LoRA for A/B/tie classification."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Local Gemma2 RM path.")
    parser.add_argument("--train", default="data/train_split.csv", help="Training CSV.")
    parser.add_argument("--valid", default="data/valid_split.csv", help="Validation CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--max-length", type=int, default=1800, help="Max token length.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of train epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Eval batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--classifier-head",
        choices=["linear", "mlp"],
        default="mlp",
        help="Classification head placed on top of Gemma2 pooled hidden states.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=0.1,
        help="Dropout used by the MLP classification head.",
    )
    parser.add_argument(
        "--head-hidden-ratio",
        type=float,
        default=0.5,
        help="MLP hidden size as a fraction of Gemma2 hidden size.",
    )
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help='Use "all-linear" or a comma-separated module list.',
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use bitsandbytes 4-bit loading. Disabled by default for 4090 bf16 LoRA.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--disable-softcapping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set Gemma2 attn/final logit softcapping config fields to None.",
    )
    parser.add_argument(
        "--valid-tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average validation predictions with A/B-flipped TTA.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Model compute dtype.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--logging-steps", type=int, default=20, help="Logging steps.")
    parser.add_argument("--eval-steps", type=int, default=200, help="Eval steps.")
    parser.add_argument("--save-steps", type=int, default=200, help="Save steps.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Max checkpoints.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional train row limit.")
    parser.add_argument("--limit-valid", type=int, default=None, help="Optional valid row limit.")
    parser.add_argument(
        "--swap-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Duplicate training rows with response A/B swapped.",
    )
    return parser.parse_args()


def predict_probabilities(trainer, dataset: PreferenceDataset) -> tuple[np.ndarray, np.ndarray]:
    prediction_output = trainer.predict(dataset)
    return softmax(prediction_output.predictions), prediction_output.label_ids


def write_validation_predictions(
    trainer,
    valid_dataset: PreferenceDataset,
    valid_swapped_dataset: PreferenceDataset | None,
    output_path: Path,
) -> float:
    probabilities, labels = predict_probabilities(trainer, valid_dataset)
    if valid_swapped_dataset is not None:
        swapped_probabilities, _ = predict_probabilities(trainer, valid_swapped_dataset)
        restored_swapped = swapped_probabilities[:, [1, 0, 2]]
        probabilities = (probabilities + restored_swapped) / 2.0

    loss = multiclass_log_loss(labels, probabilities)
    rows = []
    for row_id, probs in zip((example.row_id for example in valid_dataset.examples), probabilities):
        rows.append(
            {
                "id": row_id,
                LABEL_COLUMNS[0]: probs[0],
                LABEL_COLUMNS[1]: probs[1],
                LABEL_COLUMNS[2]: probs[2],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return loss


def main() -> None:
    from transformers import Trainer, TrainingArguments, set_seed

    args = parse_args()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "gemma2_qlora_valid_predictions.csv"
    target_modules = parse_target_modules(args.target_modules)

    print("[stage 1/5] Load tokenizer")
    tokenizer = load_tokenizer(args.model)
    print(f"  model: {args.model}")
    print(f"  max_length: {args.max_length}")
    print(f"  disable_softcapping: {args.disable_softcapping}")
    print(f"  target_modules: {target_modules}")
    print(f"  classifier_head: {args.classifier_head}")
    print(f"  head_dropout: {args.head_dropout}")
    print(f"  head_hidden_ratio: {args.head_hidden_ratio}")

    print("[stage 2/5] Build datasets")
    train_dataset = PreferenceDataset(
        csv_path=args.train,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.limit_train,
        has_labels=True,
        swap_augmentation=args.swap_augmentation,
    )
    valid_dataset = PreferenceDataset(
        csv_path=args.valid,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.limit_valid,
        has_labels=True,
        swap_augmentation=False,
    )
    valid_swapped_dataset = (
        PreferenceDataset(
            csv_path=args.valid,
            tokenizer=tokenizer,
            max_length=args.max_length,
            limit=args.limit_valid,
            has_labels=True,
            swap_inputs=True,
        )
        if args.valid_tta
        else None
    )
    print(f"  train rows: {len(train_dataset)}")
    print(f"  valid rows: {len(valid_dataset)}")
    print(f"  valid_tta: {args.valid_tta}")

    print("[stage 3/5] Load Gemma2 classifier with LoRA")
    model = load_gemma2_sequence_classifier(
        model_path=args.model,
        load_in_4bit=args.load_in_4bit,
        dtype=args.dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        gradient_checkpointing=args.gradient_checkpointing,
        disable_softcapping=args.disable_softcapping,
        classifier_head=args.classifier_head,
        head_dropout=args.head_dropout,
        head_hidden_ratio=args.head_hidden_ratio,
    )
    model.print_trainable_parameters()

    training_args = make_training_arguments(
        TrainingArguments,
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        fp16=args.dtype == "float16",
        bf16=args.dtype == "bfloat16",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForPreference(tokenizer),
        compute_metrics=trainer_compute_metrics,
    )

    print("[stage 4/5] Train")
    trainer.train()

    print("[stage 5/5] Save adapter and validation predictions")
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    (adapter_dir / "gemma2_qlora_config.json").write_text(
        "{\n"
        f'  "classifier_head": "{args.classifier_head}",\n'
        f'  "head_dropout": {args.head_dropout},\n'
        f'  "head_hidden_ratio": {args.head_hidden_ratio},\n'
        f'  "disable_softcapping": {str(args.disable_softcapping).lower()}\n'
        "}\n",
        encoding="utf-8",
    )
    valid_loss = write_validation_predictions(
        trainer,
        valid_dataset,
        valid_swapped_dataset,
        predictions_path,
    )
    print(f"log_loss={valid_loss:.8f}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {adapter_dir}")


if __name__ == "__main__":
    main()
