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
  Qwen__Qwen2.5-3B-Instruct/

scripts/
  create_validation_split.py
  evaluate_log_loss.py
  score_gemma_rm.py
  train_rm_calibrator.py

src/
  Gemma2_QLoRA/
  Qwen_QloRA/
  RM_LogisticRegression/
```

Use the `deepl` environment:

```bash
conda activate deepl
```

Current defaults are tuned for an RTX 4090 24GB GPU and use bf16 LoRA, not 4-bit quantization.

## Recommended Main Run

Gemma2 RM LoRA is the main high-quality path:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/train.py \
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

Evaluate:

```bash
python scripts/evaluate_log_loss.py \
  --predictions output/gemma2_qlora_rm/gemma2_qlora_valid_predictions.csv
```

Predict test:

```bash
PYTHONPATH=src python src/Gemma2_QLoRA/predict.py \
  --model models/sfairXC__FsfairX-Gemma2-RM-v0.1 \
  --adapter output/gemma2_qlora_rm/adapter \
  --input data/test.csv \
  --output output/gemma2_qlora_rm/submission.csv
```

If Gemma2 hits OOM, use `--batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 16`.

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

Parameters:

- `--model`: local Gemma2 RM path. Recommended: `models/sfairXC__FsfairX-Gemma2-RM-v0.1`.
- `--train`: training split. Recommended: `data/train_split.csv`.
- `--valid`: validation split. Recommended: `data/valid_split.csv`.
- `--output-dir`: output directory for checkpoints, adapter, and validation predictions.
- `--max-length`: token truncation length. Recommended: `1800`; lower to `1536` or `1024` if OOM.
- `--batch-size`: per-device train batch size. Recommended for 4090: `2`.
- `--eval-batch-size`: per-device eval batch size. Recommended for 4090: `2`.
- `--gradient-accumulation-steps`: accumulation steps. Recommended: `8`, giving effective batch `16`.
- `--epochs`: training epochs. Recommended: `1` first, then try `2`.
- `--learning-rate`: LoRA learning rate. Recommended: `2e-4`; try `1e-4` if unstable.
- `--dtype`: model dtype. Recommended: `bfloat16`.
- `--load-in-4bit`: optional quantized loading. Recommended: leave disabled on 4090.
- `--lora-r`: LoRA rank. Recommended: `64`.
- `--lora-alpha`: LoRA alpha. Recommended: `16`.
- `--lora-dropout`: LoRA dropout. Recommended: `0.05`.
- `--target-modules`: LoRA targets. Recommended: `all-linear`.
- `--classifier-head`: `mlp` or `linear`. Recommended: `mlp`.
- `--head-dropout`: MLP head dropout. Recommended: `0.1`.
- `--head-hidden-ratio`: MLP hidden size ratio. Recommended: `0.5`.
- `--disable-softcapping`: disable Gemma2 softcapping. Recommended: enabled.
- `--swap-augmentation`: duplicate training rows with A/B swapped. Recommended: enabled.
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
- `--max-length`: token truncation length. Use the training value, recommended `1800`.
- `--batch-size`: inference batch size. Recommended for 4090: `2`.
- `--has-labels`: set when predicting validation CSVs that include labels.
- `--tta`: A/B flip test-time augmentation. Recommended: enabled.
- `--dtype`: model dtype. Recommended: `bfloat16`.
- `--load-in-4bit`: optional quantized loading. Recommended: leave disabled on 4090.
- `--classifier-head`, `--head-dropout`, `--head-hidden-ratio`: loaded from adapter config by default; override only when needed.
- `--disable-softcapping`: should match training. Recommended: enabled.
- `--limit`: optional row limit for debugging. Recommended: unset for real predictions.

## Qwen LoRA Scripts

### `src/Qwen_QloRA/train.py`

Fine-tunes `Qwen2.5-3B-Instruct` as a three-class classifier. This is lighter than Gemma2 and useful as a baseline or student model.

Recommended command:

```bash
PYTHONPATH=src python src/Qwen_QloRA/train.py \
  --train data/train_split.csv \
  --valid data/valid_split.csv \
  --model models/Qwen__Qwen2.5-3B-Instruct \
  --output-dir output/qwen_qlora_3b \
  --max-length 1800 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --swap-augmentation
```

Parameters:

- `--model`: local Qwen path. Recommended: `models/Qwen__Qwen2.5-3B-Instruct`.
- `--train`, `--valid`: train and validation split CSVs.
- `--output-dir`: output directory.
- `--max-length`: token truncation length. Recommended: `1800`.
- `--batch-size`: per-device train batch. Recommended for 4090: `4`; use `2` if OOM.
- `--eval-batch-size`: per-device eval batch. Recommended for 4090: `4`.
- `--gradient-accumulation-steps`: recommended `4`, effective batch `16`.
- `--epochs`: recommended `1` first.
- `--learning-rate`: recommended `2e-4`.
- `--dtype`: recommended `bfloat16`.
- `--load-in-4bit`: optional quantized loading. Recommended: leave disabled on 4090.
- `--lora-r`: recommended `64`.
- `--lora-alpha`: recommended `16`.
- `--lora-dropout`: recommended `0.05`.
- `--target-modules`: recommended `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- `--swap-augmentation`: recommended enabled.
- `--eval-steps`, `--save-steps`: recommended `200`.
- `--limit-train`, `--limit-valid`: optional debug limits. Recommended: unset.

Output:

```text
output/qwen_qlora_3b/adapter/
output/qwen_qlora_3b/qwen_qlora_valid_predictions.csv
```

### `src/Qwen_QloRA/predict.py`

Loads a trained Qwen LoRA adapter and writes probabilities.

Recommended command:

```bash
PYTHONPATH=src python src/Qwen_QloRA/predict.py \
  --model models/Qwen__Qwen2.5-3B-Instruct \
  --adapter output/qwen_qlora_3b/adapter \
  --input data/test.csv \
  --output output/qwen_qlora_3b/submission.csv
```

Parameters:

- `--model`: base Qwen path.
- `--adapter`: trained adapter directory.
- `--input`: CSV to predict.
- `--output`: output probability CSV.
- `--max-length`: use the training value, recommended `1800`.
- `--batch-size`: inference batch size. Recommended for 4090: `4`.
- `--has-labels`: set for validation CSVs.
- `--dtype`: recommended `bfloat16`.
- `--load-in-4bit`: optional quantized loading. Recommended: leave disabled on 4090.
- `--limit`: optional debug row limit. Recommended: unset.

## RM Logistic Regression Baseline

This baseline scores each response independently with Gemma2 RM, then trains a logistic-regression calibrator on score and length features. It is cheaper to analyze and useful for ensembling, but the direct Gemma2 LoRA classifier is the stronger main path.

### `scripts/score_gemma_rm.py`

Writes score features for a split.

Recommended commands:

```bash
RUN_NAME=$(date +"%Y-%m-%d_%H-%M")

python scripts/score_gemma_rm.py \
  --input data/valid_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --dtype bfloat16 \
  --gpu-memory 23GiB \
  --save-every 100 \
  --resume

python scripts/score_gemma_rm.py \
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
- `--max-length`: token truncation length. Recommended: `512` for fast RM scoring.
- `--batch-size`: inference batch size. Recommended for 4090: `2`.
- `--dtype`: recommended `bfloat16`.
- `--gpu-memory`: max GPU memory for device map. Recommended: `23GiB`.
- `--cpu-memory`: CPU offload memory. Recommended: default `48GiB`.
- `--load-in-4bit`: optional quantized loading. Recommended: leave disabled on 4090.
- `--limit`: optional row limit. Recommended: unset for full scores.
- `--save-every`: checkpoint interval. Recommended: `100`.
- `--resume`: skip already-scored ids. Recommended: enabled.

### `scripts/train_rm_calibrator.py`

Trains a multinomial Logistic Regression calibrator on RM score files.

Recommended command:

```bash
python scripts/train_rm_calibrator.py \
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
