# LLM Preference Modeling

This project predicts which response is preferred for each prompt:

```text
winner_model_a, winner_model_b, winner_tie
```

The primary validation metric is multi-class log loss on predicted probabilities. Scripts also print hard-label accuracy from `argmax(probabilities)`.

## Layout

```text
data/
  train.csv
  test.csv
  train_split.csv
  valid_split.csv

models/
  sfairXC__FsfairX-Gemma2-RM-v0.1/

scripts/
  create_validation_split.py
  download_models.py
  download_trained_outputs.py
  evaluate_log_loss.py
  score_gemma_rm.py

src/
  EncoderEnsemble/
    data/
      processing.py
    inference/
      rule_adjustment.py
    models/
      blocks.py
    utils/
      device.py
      plotting.py
    train_pretrained_preference_model.py
    train_pairwise_context_response_encoder.py
    train_cross_context_response_encoder.py
    train_single_turn_weighted_encoder.py
    ensemble_preference_models.py
    predict_ensemble_test.py
  Gemma2_QLoRA/
    data/
      preference.py
    models/
      modeling.py
    utils/
      constants.py
      metrics.py
      representations.py
    train.py
    predict.py
    extract_gemma2_features.py
    train_gemma2_mlp_head_from_features.py
    predict_gemma2_mlp_head_from_features.py
    search_gemma_mlp_head_hparams.py
    search_gemma2_feature_mlp_head_hparams.py
  RM_LogisticRegression/
    data/
      processing.py
    inference/
      rm_scoring.py
    models/
      calibration.py
      mlp_calibration.py
      prototype_calibration.py
    utils/
      constants.py
      metrics.py
      paths.py
    train_rm_calibrator.py
    train_rm_mlp_calibrator.py
    train_rm_prototype_calibrator.py
    search_rm_mlp_hparams.py
    search_rm_prototype_hparams.py
```

Use the `deepl` environment:

```bash
conda activate deepl
```

## Prepare Models

Download all pretrained model folders used by the project:

```bash
python scripts/download_models.py \
  --hf-endpoint https://hf-mirror.com
```

The script checks `models/` first and downloads only missing or incomplete
folders. Required local model folders:

```text
models/distilbert-base-uncased
models/deberta-v3-large
models/sfairXC__FsfairX-Gemma2-RM-v0.1
```

Download trained output/checkpoint artifacts from KaggleHub:

```bash
python scripts/download_trained_outputs.py
```

Default output folders:

```text
output/gemma2_finetune/
output/EncoderEnsemble/
```

All GPU-heavy training and evaluation scripts support hardware profiles:

- `--hardware-profile auto`: default; detects dual H20 when at least two visible CUDA devices include `H20`, detects single H20 when exactly one visible CUDA device includes `H20`, otherwise uses 4090 defaults.
- `--hardware-profile 4090`: RTX 4090 24GB single-GPU defaults.
- `--hardware-profile h20`: one H20 GPU. Use `CUDA_VISIBLE_DEVICES=0` to run on the first physical GPU.
- `--hardware-profile h20x2`: two H20 GPUs. Use `torchrun --nproc_per_node 2` for LoRA training.

Profiles only fill defaults. Explicit values such as `--batch-size`, `--eval-batch-size`, `--gradient-accumulation-steps`, `--dtype`, `--load-in-4bit`, and `--gpu-memory` still override the profile.

## Four-Encoder Preference Ensemble

This method trains and ensembles four encoder classifiers for pairwise response
preference prediction:

```text
context encoder        DistilBERT, separate context/response branches
pairwise encoder       DeBERTa-v3-large, pair A and pair B branches
cross encoder          DeBERTa-v3-large, shared cross-input branch
single-turn encoder    DeBERTa-v3-large, weighted current-turn samples
```

Train the four checkpoints:

```bash
PYTHONPATH=src python src/EncoderEnsemble/train_pretrained_preference_model.py \
  --train-size 50000 \
  --valid-size 5000 \
  --epochs 2 \
  --batch-size 4 \
  --local-files-only

PYTHONPATH=src python src/EncoderEnsemble/train_pairwise_context_response_encoder.py \
  --train-size 50000 \
  --valid-size 5000 \
  --epochs 3 \
  --batch-size 1 \
  --lr 5e-7 \
  --local-files-only

PYTHONPATH=src python src/EncoderEnsemble/train_cross_context_response_encoder.py \
  --train-size 50000 \
  --valid-size 5000 \
  --epochs 3 \
  --batch-size 1 \
  --lr 5e-7 \
  --local-files-only

PYTHONPATH=src python src/EncoderEnsemble/train_single_turn_weighted_encoder.py \
  --train-size 50000 \
  --valid-size 5000 \
  --epochs 2 \
  --batch-size 1 \
  --lr 5e-7 \
  --local-files-only
```

Default checkpoint outputs:

```text
output/EncoderEnsemble/best_context_pretrained_preference_model.pt
output/EncoderEnsemble/best_pairwise_context_response_encoder.pt
output/EncoderEnsemble/best_cross_context_response_encoder.pt
output/EncoderEnsemble/best_single_turn_weighted_encoder.pt
```

Evaluate the validation ensemble:

```bash
PYTHONPATH=src python src/EncoderEnsemble/ensemble_preference_models.py \
  --train-size 50000 \
  --valid-size 5000 \
  --batch-size 8 \
  --local-files-only
```

Default ensemble weights:

```text
context      0.1845
pairwise     0.3197
cross        0.3605
single_turn  0.1354
```

Validation writes:

```text
output/EncoderEnsemble/ensemble_config.json
```

Generate test predictions:

```bash
PYTHONPATH=src python src/EncoderEnsemble/predict_ensemble_test.py \
  --batch-size 8 \
  --local-files-only \
  --output-path output/EncoderEnsemble/submission.csv
```

Useful arguments:

- `--base-path`: project root containing `data/`, `models/`, and `output/`.
- `--train-size`, `--valid-size`: sampled train/validation sizes.
- `--history-turns`: number of previous conversation turns packed into context.
- `--max-length`: tokenizer truncation length. Recommended: `512`.
- `--batch-size`: training or inference batch size.
- `--local-files-only`: use only local HuggingFace model folders.
- `--allow-cpu`: allow CPU execution for smoke tests.
- `--weights`: override ensemble weights in context, pairwise, cross, single-turn order.
- `--disable-rule-adjustment`: disable rule-based post-processing.

## RM Logistic Regression Baseline

This baseline scores each response independently with Gemma2 RM, then trains a logistic-regression calibrator on score and length features. It is cheaper to analyze and useful for ensembling, but the direct Gemma2 LoRA classifier is the stronger main path.

### `scripts/score_gemma_rm.py`

Writes score features for a split.

Recommended commands:

```bash
RUN_NAME=$(date +"%Y-%m-%d_%H-%M")

python scripts/score_gemma_rm.py \
  --hardware-profile 4090 \
  --input data/valid_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --dtype bfloat16 \
  --gpu-memory 23GiB \
  --save-every 100 \
  --resume

python scripts/score_gemma_rm.py \
  --hardware-profile 4090 \
  --input data/train_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --dtype bfloat16 \
  --gpu-memory 23GiB \
  --save-every 100 \
  --resume
```

Parameters:

- `--model`: local Gemma2 RM path.
- `--input`: split CSV to score.
- `--output`: explicit score CSV path. Usually leave unset.
- `--run-name`: output folder name. Use one shared value for train and valid scores.
- `--hardware-profile`: `auto`, `4090`, `h20`, or `h20x2`.
- `--max-length`: token truncation length. Recommended: `512` for fast RM scoring.
- `--batch-size`: inference batch size. Profile defaults: 4090=`2`, h20=`8`, h20x2=`8`.
- `--dtype`: recommended `bfloat16`.
- `--gpu-memory`: max memory per visible GPU for device map. Profile defaults: 4090=`23GiB`, h20=`90GiB`, h20x2=`90GiB`.
- `--cpu-memory`: CPU offload memory. Profile defaults: 4090=`48GiB`, h20=`96GiB`, h20x2=`96GiB`.
- `--load-in-4bit`: optional quantized loading. Recommended: leave disabled on 4090.
- `--limit`: optional row limit. Recommended: unset for full scores.
- `--save-every`: checkpoint interval. Recommended: `100`.
- `--resume`: skip already-scored ids. Recommended: enabled.

### `src/RM_LogisticRegression/train_rm_calibrator.py`

Trains a multinomial Logistic Regression calibrator on RM score files.

Recommended command:

```bash
PYTHONPATH=src python src/RM_LogisticRegression/train_rm_calibrator.py \
  --run-name $RUN_NAME
```

Parameters:

- `--train-labels`: training labels. Recommended: `data/train_split.csv`.
- `--valid-labels`: validation labels. Recommended: `data/valid_split.csv`.
- `--train-scores`: train score CSV. Usually inferred from `--run-name`.
- `--valid-scores`: valid score CSV. Usually inferred from `--run-name`.
- `--output`: validation probability output CSV. Usually inferred from `--run-name`.
- `--model-output`: saved calibrator path. Usually inferred from `--run-name`.
- `--run-name`: output run folder shared with `score_gemma_rm.py`.
- `--c`: inverse regularization strength. Recommended: `1.0`; try `0.3`, `1`, `3`.
- `--max-iter`: optimizer iterations. Recommended: `2000`.
- `--allow-partial`: train/evaluate on matched scored ids only. Recommended only for partial/debug scores.
- `--augment-swapped` / `--no-augment-swapped`: duplicate training rows with A/B swapped. Enabled by default.
- `--shuffle-seed`: seed used to shuffle original and swapped rows together. Recommended: `42`.

The calibrators read the original RM score columns from disk, then add derived
features in memory:

```text
score_sum, score_mean, score_product, score_max, score_min
response_len_abs_diff, response_len_sum, response_len_mean, response_len_ratio
log_prompt_len, log_response_a_len, log_response_b_len
```

Swapped augmentation creates one reversed training row for each original row:

```text
score_a <-> score_b
response_a_len <-> response_b_len
score_diff -> -score_diff
response_len_diff -> -response_len_diff
winner_model_a <-> winner_model_b
winner_tie unchanged
```

The original and reversed rows are shuffled after concatenation, so opposite pairs are not placed next to each other during training.

Outputs:

```text
output/<RUN_NAME>/RM_LogisticRegression/rm_calibrated_valid_predictions.csv
output/<RUN_NAME>/RM_LogisticRegression/rm_logistic_regression_model.joblib
```

The script prints:

```text
log_loss
manual_log_loss
accuracy
rows
```

### `src/RM_LogisticRegression/train_rm_mlp_calibrator.py`

Trains a small MLP calibrator on the same RM score and length features. This is
separate from the Logistic Regression baseline, uses AdamW weight decay, and
stops early when validation loss stops improving.

Recommended command:

```bash
PYTHONPATH=src python src/RM_LogisticRegression/train_rm_mlp_calibrator.py \
  --run-name $RUN_NAME \
  --hidden-dim 32 \
  --dropout 0.1 \
  --learning-rate 1e-3 \
  --weight-decay 1e-2 \
  --batch-size 512 \
  --max-epochs 300 \
  --patience 5
```

Additional parameters:

- `--min-delta`: minimum validation-loss improvement to reset early stopping.
- `--device`: `auto`, `cpu`, `cuda`, or a CUDA device such as `cuda:0`.

Outputs:

```text
output/<RUN_NAME>/RM_LogisticRegression/rm_mlp_calibrated_valid_predictions.csv
output/<RUN_NAME>/RM_LogisticRegression/rm_mlp_calibrator.pt
```

### `src/RM_LogisticRegression/search_rm_mlp_hparams.py`

Runs a grid search over MLP calibrator hyperparameters and saves the best
validation predictions/checkpoint.

Recommended quick search:

```bash
PYTHONPATH=src python src/RM_LogisticRegression/search_rm_mlp_hparams.py \
  --run-name $RUN_NAME \
  --hidden-dims 16,32,64 \
  --dropouts 0.05,0.1 \
  --learning-rates 0.0005,0.001 \
  --weight-decays 0.001,0.003,0.01 \
  --batch-sizes 512 \
  --max-epochs 500 \
  --patience 10
```

Use a smaller grid for debugging:

```bash
PYTHONPATH=src python src/RM_LogisticRegression/search_rm_mlp_hparams.py \
  --run-name $RUN_NAME \
  --hidden-dims 32 \
  --dropouts 0.1 \
  --learning-rates 0.001 \
  --weight-decays 0.003 \
  --batch-sizes 512 \
  --max-epochs 20 \
  --patience 3
```

Outputs:

```text
output/<RUN_NAME>/RM_LogisticRegression/rm_mlp_hparam_search_results.csv
output/<RUN_NAME>/RM_LogisticRegression/rm_mlp_hparam_best_valid_predictions.csv
output/<RUN_NAME>/RM_LogisticRegression/rm_mlp_hparam_best_model.pt
```

## Gemma2 LoRA Scripts

### `src/Gemma2_QLoRA/train.py`

Fine-tunes `sfairXC__FsfairX-Gemma2-RM-v0.1` directly as a three-class preference classifier.

Input format:

```text
<PROMPT>...</PROMPT><RESPONSE A>...</RESPONSE A><RESPONSE B>...</RESPONSE B>
```

Default classification head:

```text
LayerNorm(3584)
Dropout(0.1)
Linear(3584, 1792)
GELU
Dropout(0.1)
Linear(1792, 3)
```

Recommended command:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/train.py \
  --hardware-profile 4090 \
  --train data/train_split.csv \
  --valid data/valid_split.csv \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --output-dir output/gemma2_qlora_rm \
  --max-length 1800 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --learning-rate 2e-4
```

Representation-aware run with A/B swap consistency and online class prototypes:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/train.py \
  --hardware-profile 4090 \
  --train data/train_split.csv \
  --valid data/valid_split.csv \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --output-dir output/gemma2_qlora_rm_repr \
  --max-length 1800 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --swap-consistency-weight 0.05 \
  --prototype-loss-weight 0.02
```

Train only the MLP classification head from an existing representation-aware adapter
while freezing the base model and all LoRA weights:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/train.py \
  --hardware-profile 4090 \
  --train data/train_split.csv \
  --valid data/valid_split.csv \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --resume-adapter output/gemma2_qlora_rm_rep/adapter \
  --output-dir output/gemma2_qlora_rm_rep_head_only \
  --max-length 1800 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --learning-rate 1e-3 \
  --classifier-head mlp \
  --head-dropout 0.1 \
  --head-hidden-ratio 0.5 \
  --head-only \
  --swap-consistency-weight 0.05 \
  --prototype-loss-weight 0.0
```

For much faster MLP-head tuning, first extract pooled hidden features once with the
frozen Gemma2+LoRA model, then train the head from those saved tensors:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/extract_gemma2_features.py \
  --hardware-profile 4090 \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --adapter output/gemma2_qlora_rm_rep/adapter \
  --input data/train_split.csv \
  --output output/gemma2_qlora_rm_rep_features/train_features.pt \
  --max-length 1800 \
  --has-labels \
  --swap-augmentation

PYTHONPATH=src python src/Gemma2_QLoRA/extract_gemma2_features.py \
  --hardware-profile 4090 \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --adapter output/gemma2_qlora_rm_rep/adapter \
  --input data/valid_split.csv \
  --output output/gemma2_qlora_rm_rep_features/valid_features.pt \
  --max-length 1800 \
  --has-labels

PYTHONPATH=src python src/Gemma2_QLoRA/train_gemma2_mlp_head_from_features.py \
  --train-features output/gemma2_qlora_rm_rep_features/train_features.pt \
  --valid-features output/gemma2_qlora_rm_rep_features/valid_features.pt \
  --output-dir output/gemma2_qlora_rm_rep_feature_head \
  --head-dropout 0.1 \
  --head-hidden-ratio 0.5 \
  --epochs 30 \
  --batch-size 256 \
  --learning-rate 1e-3
```

Grid-search the offline MLP head without rerunning Gemma2:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/search_gemma2_feature_mlp_head_hparams.py \
  --train-features output/gemma2_qlora_rm_rep_features/train_features.pt \
  --valid-features output/gemma2_qlora_rm_rep_features/valid_features.pt \
  --search-output-dir output/gemma2_qlora_rm_rep_feature_head_search \
  --dropouts 0.0,0.05,0.1,0.15,0.2 \
  --hidden-ratios 0.25,0.5,0.75,1.0 \
  --learning-rates 3e-4,1e-3,3e-3 \
  --weight-decays 0.0,0.01 \
  --epochs 30 \
  --batch-size 256 \
  --patience 5
```

To predict from a saved head checkpoint, extract features for the target CSV and
run the standalone head:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/extract_gemma2_features.py \
  --hardware-profile 4090 \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --adapter output/gemma2_qlora_rm_rep/adapter \
  --input data/test.csv \
  --output output/gemma2_qlora_rm_rep_features/test_features.pt \
  --max-length 1800

PYTHONPATH=src python src/Gemma2_QLoRA/predict_gemma2_mlp_head_from_features.py \
  --features output/gemma2_qlora_rm_rep_features/test_features.pt \
  --checkpoint output/gemma2_qlora_rm_rep_feature_head/gemma2_mlp_head.pt \
  --output output/gemma2_qlora_rm_rep_feature_head/submission.csv
```

Search MLP head hyperparameters in head-only mode:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/search_gemma_mlp_head_hparams.py \
  --hardware-profile 4090 \
  --train data/train_split.csv \
  --valid data/valid_split.csv \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --resume-adapter output/gemma2_qlora_rm_rep/adapter \
  --search-output-dir output/gemma2_qlora_rm_rep_head_only_search \
  --dropouts 0.0,0.05,0.1,0.15,0.2 \
  --hidden-ratios 0.25,0.5,0.75,1.0 \
  --epochs 1 \
  --learning-rate 1e-3 \
  --prototype-loss-weight 0.0 \
  --head-only
```

Continue from a previous trained adapter for another fine-tuning round:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/train.py \
  --hardware-profile 4090 \
  --train data/train_split.csv \
  --valid data/valid_split.csv \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --resume-adapter output/gemma2_qlora_rm/adapter \
  --output-dir output/gemma2_qlora_rm_round2 \
  --max-length 1800 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --learning-rate 5e-5
```

Parameters:

- `--model`: local Gemma2 RM path. Recommended: `models/sfairXC__FsfairX-Gemma2-RM-v0.1`.
- `--train`: training split. Recommended: `data/train_split.csv`.
- `--valid`: validation split. Recommended: `data/valid_split.csv`.
- `--output-dir`: output directory for checkpoints, adapter, and validation predictions.
- `--resume-adapter`: optional previous `adapter/` directory to initialize from for another fine-tuning round. Keep `--model` pointed at the original base model and write to a new `--output-dir`.
- `--hardware-profile`: `auto`, `4090`, `h20`, or `h20x2`. Recommended: `auto`; prefix the command with `CUDA_VISIBLE_DEVICES=0` and pass `--hardware-profile h20` for first-card single H20, or use `h20x2` with `torchrun --nproc_per_node 2`.
- `--max-length`: token truncation length. Recommended: `1800`; lower to `1536` or `1024` if OOM.
- `--batch-size`: per-device train batch size. Profile defaults: 4090=`2`, h20=`8`, h20x2=`8`.
- `--eval-batch-size`: per-device eval batch size. Profile defaults: 4090=`2`, h20=`8`, h20x2=`8`.
- `--gradient-accumulation-steps`: profile defaults: 4090=`8`, h20=`2`, h20x2=`2`, giving effective batch `16` on 4090/single H20 and `32` on H20 dual GPU.
- `--epochs`: training epochs. Recommended: `1` first, then try `2`.
- `--learning-rate`: LoRA learning rate. Recommended: `2e-4`; try `1e-4` if unstable.
- `--dtype`: model dtype. Recommended: `bfloat16`.
- `--load-in-4bit`: optional quantized loading. Profile default: disabled. In torchrun, 4-bit loading is pinned to each process's `LOCAL_RANK`.
- `--lora-r`: LoRA rank. Recommended: `64`.
- `--lora-alpha`: LoRA alpha. Recommended: `16`.
- `--lora-dropout`: LoRA dropout. Recommended: `0.05`.
- `--target-modules`: LoRA targets. Recommended: `all-linear`.
- `--classifier-head`: `mlp` or `linear`. Recommended: `mlp`.
- `--head-dropout`: MLP head dropout. Recommended: `0.1`.
- `--head-hidden-ratio`: MLP hidden size ratio. Recommended: `0.5`.
- `--head-only`: freeze the base model and LoRA weights; train only the classification head. Recommended when tuning the MLP head on top of an existing `gemma2_qlora_rm_rep` adapter.
- `--disable-softcapping`: disable Gemma2 softcapping. Recommended: enabled.
- `--swap-augmentation`: duplicate training rows with A/B swapped. Recommended: enabled.
- `--swap-consistency-weight`: symmetric KL loss weight between original predictions and A/B-swapped predictions after restoring A/B columns. Recommended first try: `0.03` to `0.08`.
- `--swap-ce-weight`: optional extra CE loss on swapped inputs with swapped labels. Recommended first try: `0.0`; use only if consistency alone is weak.
- `--prototype-loss-weight`: cosine center-loss weight that pulls pooled hidden states toward online A/B/tie class prototypes. Recommended first try: `0.01` to `0.03`.
- `--prototype-momentum`: EMA momentum for online class prototypes. Recommended: `0.95`.
- `--valid-tta`: average validation predictions with A/B-flipped TTA. Recommended: enabled.
- `--eval-steps`: validation frequency. Recommended: `200`.
- `--save-steps`: checkpoint frequency. Recommended: `200`.
- `--save-total-limit`: max checkpoints to keep. Recommended: `2`.
- `--limit-train`, `--limit-valid`: optional row limits for debugging. Recommended: unset for real runs.

Outputs:

```text
output/gemma2_qlora_rm/adapter/
output/gemma2_qlora_rm/gemma2_qlora_valid_predictions.csv
```

### `src/Gemma2_QLoRA/predict.py`

Loads a trained Gemma2 LoRA adapter and writes probabilities for validation or test rows.

Recommended command:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/predict.py \
  --hardware-profile 4090 \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --adapter output/gemma2_qlora_rm/adapter \
  --input data/test.csv \
  --output output/gemma2_qlora_rm/submission.csv
```

Parameters:

- `--model`: base Gemma2 RM path.
- `--adapter`: trained adapter directory. Recommended: `output/gemma2_qlora_rm/adapter`.
- `--input`: CSV to predict. Use `data/test.csv` for submission.
- `--output`: output probability CSV.
- `--hardware-profile`: `auto`, `4090`, `h20`, or `h20x2`.
- `--max-length`: token truncation length. Use the training value, recommended `1800`.
- `--batch-size`: inference batch size. Profile defaults: 4090=`2`, h20=`8`, h20x2=`8`.
- `--has-labels`: set when predicting validation CSVs that include labels.
- `--tta`: A/B flip test-time augmentation. Recommended: enabled.
- `--dtype`: model dtype. Recommended: `bfloat16`.
- `--load-in-4bit`: optional quantized loading. Recommended: leave disabled on 4090.
- `--classifier-head`, `--head-dropout`, `--head-hidden-ratio`: loaded from adapter config by default; override only when needed.
- `--disable-softcapping`: should match training. Recommended: enabled.
- `--limit`: optional row limit for debugging. Recommended: unset for real predictions.

## Data And Evaluation Scripts

### `scripts/create_validation_split.py`

Creates a stratified train/validation split.

Recommended command:

```bash
python scripts/create_validation_split.py \
  --input data/train.csv \
  --train-output data/train_split.csv \
  --valid-output data/valid_split.csv \
  --valid-size 0.1 \
  --seed 42
```

Parameters:

- `--input`: source labeled CSV. Recommended: `data/train.csv`.
- `--train-output`: output training split. Recommended: `data/train_split.csv`.
- `--valid-output`: output validation split. Recommended: `data/valid_split.csv`.
- `--valid-size`: validation fraction. Recommended: `0.1`.
- `--seed`: random seed for reproducible split. Recommended: `42`.

### `scripts/evaluate_log_loss.py`

Evaluates a prediction CSV against validation labels. It prints both log loss and accuracy.

Recommended command:

```bash
python scripts/evaluate_log_loss.py \
  --labels data/valid_split.csv \
  --predictions output/gemma2_qlora_rm/gemma2_qlora_valid_predictions.csv
```

Prediction CSV format:

```text
id,winner_model_a,winner_model_b,winner_tie
```

Parameters:

- `--labels`: validation CSV with one-hot labels. Recommended: `data/valid_split.csv`.
- `--predictions`: prediction probability CSV to evaluate.
- `--normalize`: normalize prediction rows to sum to 1 before scoring. Recommended only if your output is unnormalized.
- `--allow-partial`: evaluate matched ids only. Recommended only for partial/debug predictions.
- `--clip`: probability clipping epsilon. Recommended: default `1e-15`.
