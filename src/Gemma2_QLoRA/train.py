import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from hardware_profiles import add_hardware_profile_argument, apply_profile_defaults
from Gemma2_QLoRA.utils.constants import DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, LABEL_COLUMNS
from Gemma2_QLoRA.data.preference import DataCollatorForPreference, PreferenceDataset
from Gemma2_QLoRA.utils.metrics import multiclass_log_loss, softmax
from Gemma2_QLoRA.models.modeling import (
    load_gemma2_sequence_classifier,
    load_tokenizer,
    parse_target_modules,
)
from Gemma2_QLoRA.utils.representations import pooled_last_hidden


def make_training_arguments(training_arguments_cls, **kwargs):
    import inspect

    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return training_arguments_cls(**kwargs)


def make_trainer_processor_kwargs(trainer_cls, tokenizer):
    import inspect

    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def trainer_compute_metrics(eval_pred) -> dict[str, float]:
    logits = getattr(eval_pred, "predictions", eval_pred[0])
    labels = getattr(eval_pred, "label_ids", eval_pred[1])
    return {"log_loss": multiclass_log_loss(labels, softmax(logits))}


def swap_label_ids(labels: torch.Tensor) -> torch.Tensor:
    swapped = labels.clone()
    swapped = torch.where(labels == 0, torch.ones_like(swapped), swapped)
    swapped = torch.where(labels == 1, torch.zeros_like(swapped), swapped)
    return swapped


def restore_swapped_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, [1, 0, 2]]


class RepresentationPreferenceTrainer:
    def __init__(
        self,
        *,
        swap_consistency_weight: float,
        swap_ce_weight: float,
        prototype_loss_weight: float,
        prototype_momentum: float,
    ) -> None:
        self.swap_consistency_weight = swap_consistency_weight
        self.swap_ce_weight = swap_ce_weight
        self.prototype_loss_weight = prototype_loss_weight
        self.prototype_momentum = prototype_momentum
        self.class_prototypes = None
        self.prototype_initialized = None

    def _ensure_prototypes(self, hidden: torch.Tensor) -> None:
        if self.class_prototypes is not None:
            return
        self.class_prototypes = torch.zeros(
            len(LABEL_COLUMNS),
            hidden.size(-1),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        self.prototype_initialized = torch.zeros(
            len(LABEL_COLUMNS),
            device=hidden.device,
            dtype=torch.bool,
        )

    def _prototype_loss(
        self,
        hidden: torch.Tensor,
        labels: torch.Tensor,
        update_prototypes: bool,
    ) -> torch.Tensor:
        self._ensure_prototypes(hidden)
        hidden_norm = F.normalize(hidden.float(), dim=-1)
        prototypes = self.class_prototypes.to(device=hidden.device, dtype=torch.float32)
        initialized = self.prototype_initialized.to(device=hidden.device)
        valid = initialized[labels]

        if valid.any():
            target = prototypes[labels[valid]]
            loss = 1.0 - (hidden_norm[valid] * target).sum(dim=-1)
            prototype_loss = loss.mean()
        else:
            prototype_loss = hidden_norm.new_zeros(())

        if update_prototypes:
            with torch.no_grad():
                for class_id in range(len(LABEL_COLUMNS)):
                    class_mask = labels == class_id
                    if not class_mask.any():
                        continue
                    class_mean = F.normalize(hidden_norm[class_mask].mean(dim=0), dim=0)
                    if initialized[class_id]:
                        updated = (
                            self.prototype_momentum * prototypes[class_id]
                            + (1.0 - self.prototype_momentum) * class_mean
                        )
                        prototypes[class_id] = F.normalize(updated, dim=0)
                    else:
                        prototypes[class_id] = class_mean
                        initialized[class_id] = True
                self.class_prototypes.copy_(prototypes.to(dtype=self.class_prototypes.dtype))
                self.prototype_initialized.copy_(initialized)

        return prototype_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        swap_input_ids = inputs.pop("swap_input_ids", None)
        swap_attention_mask = inputs.pop("swap_attention_mask", None)

        need_hidden = self.prototype_loss_weight > 0.0
        outputs = model(**inputs, output_hidden_states=need_hidden)
        loss = outputs.loss
        metrics = {}

        if (
            (self.swap_consistency_weight > 0.0 or self.swap_ce_weight > 0.0)
            and swap_input_ids is not None
            and swap_attention_mask is not None
        ):
            swapped_outputs = model(
                input_ids=swap_input_ids,
                attention_mask=swap_attention_mask,
            )
            if self.swap_consistency_weight > 0.0:
                restored_swapped_logits = restore_swapped_logits(swapped_outputs.logits)
                original_log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
                original_probs = original_log_probs.exp()
                restored_log_probs = F.log_softmax(restored_swapped_logits.float(), dim=-1)
                restored_probs = restored_log_probs.exp()
                consistency_loss = 0.5 * (
                    F.kl_div(original_log_probs, restored_probs, reduction="batchmean")
                    + F.kl_div(restored_log_probs, original_probs, reduction="batchmean")
                )
                loss = loss + self.swap_consistency_weight * consistency_loss
                metrics["swap_consistency_loss"] = consistency_loss.detach()

            if self.swap_ce_weight > 0.0 and labels is not None:
                swapped_ce_loss = F.cross_entropy(
                    swapped_outputs.logits.float(),
                    swap_label_ids(labels).to(swapped_outputs.logits.device),
                )
                loss = loss + self.swap_ce_weight * swapped_ce_loss
                metrics["swap_ce_loss"] = swapped_ce_loss.detach()

        if self.prototype_loss_weight > 0.0 and labels is not None:
            hidden = pooled_last_hidden(outputs, inputs["attention_mask"])
            prototype_loss = self._prototype_loss(
                hidden,
                labels.to(hidden.device),
                update_prototypes=model.training,
            )
            loss = loss + self.prototype_loss_weight * prototype_loss
            metrics["prototype_loss"] = prototype_loss.detach()

        if metrics:
            self.log({key: value.item() for key, value in metrics.items()})

        if return_outputs:
            return loss, {"logits": outputs.logits}
        return loss


def make_preference_trainer_cls(trainer_cls):
    class Gemma2PreferenceTrainer(RepresentationPreferenceTrainer, trainer_cls):
        def __init__(
            self,
            *args,
            swap_consistency_weight: float,
            swap_ce_weight: float,
            prototype_loss_weight: float,
            prototype_momentum: float,
            **kwargs,
        ):
            RepresentationPreferenceTrainer.__init__(
                self,
                swap_consistency_weight=swap_consistency_weight,
                swap_ce_weight=swap_ce_weight,
                prototype_loss_weight=prototype_loss_weight,
                prototype_momentum=prototype_momentum,
            )
            trainer_cls.__init__(self, *args, **kwargs)

    return Gemma2PreferenceTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma2 RM with LoRA for A/B/tie classification."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Local Gemma2 RM path.")
    parser.add_argument("--train", default="data/train_split.csv", help="Training CSV.")
    parser.add_argument("--valid", default="data/valid_split.csv", help="Validation CSV.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument(
        "--resume-adapter",
        default=None,
        help="Optional trained LoRA adapter directory to initialize from for another round.",
    )
    parser.add_argument("--max-length", type=int, default=1800, help="Max token length.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of train epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    add_hardware_profile_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Eval batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
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
        default=None,
        help="Classification head placed on top of Gemma2 pooled hidden states.",
    )
    parser.add_argument(
        "--head-dropout",
        type=float,
        default=None,
        help="Dropout used by the MLP classification head.",
    )
    parser.add_argument(
        "--head-hidden-ratio",
        type=float,
        default=None,
        help="MLP hidden size as a fraction of Gemma2 hidden size.",
    )
    parser.add_argument(
        "--head-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the base model and LoRA weights; train only the classification head.",
    )
    parser.add_argument(
        "--target-modules",
        default="all-linear",
        help='Use "all-linear" or a comma-separated module list.',
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
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
        default=None,
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
    parser.add_argument(
        "--swap-consistency-weight",
        type=float,
        default=0.0,
        help="Symmetric KL weight between original predictions and restored A/B-swapped predictions.",
    )
    parser.add_argument(
        "--swap-ce-weight",
        type=float,
        default=0.0,
        help="Optional CE weight on A/B-swapped inputs with swapped labels.",
    )
    parser.add_argument(
        "--prototype-loss-weight",
        type=float,
        default=0.0,
        help="Cosine center-loss weight that pulls pooled hidden states toward class prototypes.",
    )
    parser.add_argument(
        "--prototype-momentum",
        type=float,
        default=0.95,
        help="EMA momentum for online class prototypes.",
    )
    return apply_resume_adapter_config_defaults(
        apply_profile_defaults(parser.parse_args(), "gemma_train")
    )


def apply_resume_adapter_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    saved_config = {}
    if args.resume_adapter is not None:
        config_path = Path(args.resume_adapter) / "gemma2_qlora_config.json"
        if config_path.exists():
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.classifier_head is None:
        args.classifier_head = saved_config.get("classifier_head", "mlp")
    if args.head_dropout is None:
        args.head_dropout = float(saved_config.get("head_dropout", 0.1))
    if args.head_hidden_ratio is None:
        args.head_hidden_ratio = float(saved_config.get("head_hidden_ratio", 0.5))
    return args


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
    print(f"  resume_adapter: {args.resume_adapter}")
    print(f"  max_length: {args.max_length}")
    print(f"  hardware_profile: {args.hardware_profile} ({args.hardware_description})")
    print(f"  batch_size: {args.batch_size}")
    print(f"  eval_batch_size: {args.eval_batch_size}")
    print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"  dtype: {args.dtype}")
    print(f"  load_in_4bit: {args.load_in_4bit}")
    print(f"  disable_softcapping: {args.disable_softcapping}")
    print(f"  target_modules: {target_modules}")
    print(f"  classifier_head: {args.classifier_head}")
    print(f"  head_dropout: {args.head_dropout}")
    print(f"  head_hidden_ratio: {args.head_hidden_ratio}")
    print(f"  head_only: {args.head_only}")
    print(f"  swap_consistency_weight: {args.swap_consistency_weight}")
    print(f"  swap_ce_weight: {args.swap_ce_weight}")
    print(f"  prototype_loss_weight: {args.prototype_loss_weight}")
    print(f"  prototype_momentum: {args.prototype_momentum}")
    if args.head_only and args.prototype_loss_weight > 0.0:
        print("  warning: prototype_loss_weight has no trainable hidden-state path in head-only mode.")

    print("[stage 2/5] Build datasets")
    train_dataset = PreferenceDataset(
        csv_path=args.train,
        tokenizer=tokenizer,
        max_length=args.max_length,
        limit=args.limit_train,
        has_labels=True,
        swap_augmentation=args.swap_augmentation,
        include_swapped_features=args.swap_consistency_weight > 0.0 or args.swap_ce_weight > 0.0,
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
        resume_adapter=args.resume_adapter,
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
        head_only=args.head_only,
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
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    trainer_cls = make_preference_trainer_cls(Trainer)
    trainer = trainer_cls(
        swap_consistency_weight=args.swap_consistency_weight,
        swap_ce_weight=args.swap_ce_weight,
        prototype_loss_weight=args.prototype_loss_weight,
        prototype_momentum=args.prototype_momentum,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForPreference(tokenizer),
        compute_metrics=trainer_compute_metrics,
        **make_trainer_processor_kwargs(Trainer, tokenizer),
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
        f'  "head_only": {str(args.head_only).lower()},\n'
        f'  "disable_softcapping": {str(args.disable_softcapping).lower()},\n'
        f'  "swap_consistency_weight": {args.swap_consistency_weight},\n'
        f'  "swap_ce_weight": {args.swap_ce_weight},\n'
        f'  "prototype_loss_weight": {args.prototype_loss_weight},\n'
        f'  "prototype_momentum": {args.prototype_momentum},\n'
        f'  "hardware_profile": "{args.hardware_profile}"\n'
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
