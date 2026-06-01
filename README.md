# LLM Preference Modeling

This project predicts which LLM response is preferred for a prompt:

- `winner_model_a`
- `winner_model_b`
- `winner_tie`

The validation metric is multi-class log loss between predicted probabilities and the one-hot ground truth labels.

## Current Method: RM_LogisticRegression

`RM_LogisticRegression` is a lightweight baseline with two stages:

1. Use `sfairXC/FsfairX-Gemma2-RM-v0.1` as a reward model to score each response independently.
2. Train a multinomial Logistic Regression calibrator on reward scores and length features.

For each row:

```text
score_a = RM(prompt, response_a)
score_b = RM(prompt, response_b)
```

The calibrator learns:

```text
[score_a, score_b, score_diff, score_abs_diff, length features]
  -> [P(A wins), P(B wins), P(tie)]
```

This is useful because the reward model gives a quality score for one response, while the competition needs calibrated three-class probabilities.

## Project Layout

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  train_split.csv
  valid_split.csv

models/
  sfairXC__FsfairX-Gemma2-RM-v0.1/
  Qwen__Qwen2.5-3B-Instruct/

output/
  YYYY-MM-DD_HH-MM/
    RM_LogisticRegression/
      train_split_gemma_rm_scores.csv
      valid_split_gemma_rm_scores.csv
      rm_calibrated_valid_predictions.csv
      rm_logistic_regression_model.joblib

scripts/
  create_validation_split.py
  score_gemma_rm.py
  train_rm_calibrator.py
  evaluate_log_loss.py

src/
  RM_LogisticRegression/
```

`data/` should hold raw data and reusable splits. Model outputs and experiment artifacts should go under `output/<run-name>/<method-name>/`.

## Environment

Use the `deepl` conda environment:

```bash
conda activate deepl
```

Or run commands without activating:

```bash
conda run -n deepl python ...
```

The Gemma2 reward model can run on an 8GB GPU with 4-bit loading. Recommended starting settings:

```text
max_length=512
batch_size=2
gpu_memory=7.6GiB
dtype=float16
```

If you hit OOM, lower `--batch-size` to `1`, then lower `--max-length` to `384` or `256`.

## Run Name

Outputs are grouped by run name:

```text
output/YYYY-MM-DD_HH-MM/RM_LogisticRegression/
```

By default, scripts use the current time down to the minute, for example:

```text
output/2026-06-01_00-20/RM_LogisticRegression/
```

For a full experiment, set one run name and reuse it in every command:

```bash
RUN_NAME=2026-06-01_00-20
```

This matters because reward scoring can take a long time, and the calibrator needs to read the score files from the same run folder.

## Step 1: Create Validation Split

Script:

```text
scripts/create_validation_split.py
```

Purpose:

- Reads `data/train.csv`
- Creates a stratified train/validation split
- Preserves the A/B/tie label distribution
- Writes `data/train_split.csv` and `data/valid_split.csv`

Default command:

```bash
python scripts/create_validation_split.py
```

Equivalent explicit command:

```bash
python scripts/create_validation_split.py \
  --input data/train.csv \
  --train-output data/train_split.csv \
  --valid-output data/valid_split.csv \
  --valid-size 0.1 \
  --seed 42
```

Output:

```text
data/train_split.csv
data/valid_split.csv
```

## Step 2: Score Responses With Gemma2 RM

Script:

```text
scripts/score_gemma_rm.py
```

Purpose:

- Loads the local Gemma2 reward model
- Scores `prompt + response_a`
- Scores `prompt + response_b`
- Writes score and length features to CSV

Important output columns:

```text
id
score_a
score_b
score_diff
score_abs_diff
prompt_len
response_a_len
response_b_len
response_len_diff
```

### Smoke Test

Run a tiny test first:

```bash
RUN_NAME=2026-06-01_00-20

conda run -n deepl python scripts/score_gemma_rm.py \
  --input data/valid_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --load-in-4bit \
  --dtype float16 \
  --gpu-memory 7.6GiB \
  --limit 10 \
  --save-every 5 \
  --resume
```

This writes:

```text
output/<RUN_NAME>/RM_LogisticRegression/valid_split_gemma_rm_scores_limit10.csv
```

### Score Validation Split

```bash
RUN_NAME=2026-06-01_00-20

conda run -n deepl python scripts/score_gemma_rm.py \
  --input data/valid_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --load-in-4bit \
  --dtype float16 \
  --gpu-memory 7.6GiB \
  --save-every 100 \
  --resume
```

Default output:

```text
output/<RUN_NAME>/RM_LogisticRegression/valid_split_gemma_rm_scores.csv
```

### Score Training Split

For a quick partial run:

```bash
RUN_NAME=2026-06-01_00-20

conda run -n deepl python scripts/score_gemma_rm.py \
  --input data/train_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --load-in-4bit \
  --dtype float16 \
  --gpu-memory 7.6GiB \
  --limit 1000 \
  --save-every 100 \
  --resume
```

Default output:

```text
output/<RUN_NAME>/RM_LogisticRegression/train_split_gemma_rm_scores_limit1000.csv
```

For full training scores, remove `--limit`:

```bash
RUN_NAME=2026-06-01_00-20

conda run -n deepl python scripts/score_gemma_rm.py \
  --input data/train_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --load-in-4bit \
  --dtype float16 \
  --gpu-memory 7.6GiB \
  --save-every 100 \
  --resume
```

Default output:

```text
output/<RUN_NAME>/RM_LogisticRegression/train_split_gemma_rm_scores.csv
```

`--resume` skips ids that already exist in the output CSV, so long scoring jobs can be stopped and restarted.

## Step 3: Train Logistic Regression Calibrator

Script:

```text
scripts/train_rm_calibrator.py
```

Purpose:

- Reads train labels
- Reads validation labels
- Reads RM score CSV files
- Trains a multinomial Logistic Regression calibrator
- Writes calibrated validation probabilities
- Saves the trained classifier model
- Prints validation log loss

### Full Default Run

This expects the full default score files:

```text
output/<RUN_NAME>/RM_LogisticRegression/train_split_gemma_rm_scores.csv
output/<RUN_NAME>/RM_LogisticRegression/valid_split_gemma_rm_scores.csv
```

Command:

```bash
RUN_NAME=2026-06-01_00-20

python scripts/train_rm_calibrator.py \
  --run-name $RUN_NAME
```

Output:

```text
output/<RUN_NAME>/RM_LogisticRegression/rm_calibrated_valid_predictions.csv
output/<RUN_NAME>/RM_LogisticRegression/rm_logistic_regression_model.joblib
```

### Partial Run

If you only scored part of the train or validation split, use `--allow-partial` and pass the score files explicitly:

```bash
RUN_NAME=2026-06-01_00-20

python scripts/train_rm_calibrator.py \
  --train-scores output/$RUN_NAME/RM_LogisticRegression/train_split_gemma_rm_scores_limit1000.csv \
  --valid-scores output/$RUN_NAME/RM_LogisticRegression/valid_split_gemma_rm_scores.csv \
  --output output/$RUN_NAME/RM_LogisticRegression/rm_calibrated_valid_predictions_partial.csv \
  --model-output output/$RUN_NAME/RM_LogisticRegression/rm_logistic_regression_model_limit1000.joblib \
  --allow-partial
```

Current partial result:

```text
1000 train score rows
1300 valid score rows
log_loss ~= 1.04195
```

Uniform baseline is:

```text
log_loss ~= 1.09861
```

So the RM scores are already useful.

## Step 4: Evaluate Predictions

Script:

```text
scripts/evaluate_log_loss.py
```

Purpose:

- Reads a predictions CSV
- Joins it with validation labels by `id`
- Computes multi-class log loss

Prediction CSV format:

```text
id,winner_model_a,winner_model_b,winner_tie
```

Command:

```bash
RUN_NAME=2026-06-01_00-20

python scripts/evaluate_log_loss.py \
  --predictions output/$RUN_NAME/RM_LogisticRegression/rm_calibrated_valid_predictions.csv
```

For partial validation predictions:

```bash
RUN_NAME=2026-06-01_00-20

python scripts/evaluate_log_loss.py \
  --predictions output/$RUN_NAME/RM_LogisticRegression/rm_calibrated_valid_predictions_partial.csv \
  --allow-partial
```

Useful options:

- `--labels`: label CSV, default `data/valid_split.csv`
- `--normalize`: normalize prediction rows before evaluation
- `--allow-partial`: evaluate matched ids only
- `--clip`: probability clipping value before log loss

## Script Summary

### `scripts/create_validation_split.py`

Creates reproducible stratified train/validation splits.

Key args:

- `--input`
- `--train-output`
- `--valid-output`
- `--valid-size`
- `--seed`

### `scripts/score_gemma_rm.py`

Runs Gemma2 RM inference and writes score features.

Key args:

- `--input`: CSV to score
- `--output`: explicit output path
- `--run-name`: output folder name, e.g. `2026-06-01_00-20`
- `--limit`: optional row limit for smoke tests
- `--resume`: skip ids already written
- `--max-length`: token truncation length
- `--batch-size`: inference batch size
- `--load-in-4bit`: 4-bit loading, enabled by default
- `--dtype`: usually `float16`
- `--gpu-memory`: recommended `7.6GiB` for an 8GB GPU

### `scripts/train_rm_calibrator.py`

Trains Logistic Regression on RM score features.

Key args:

- `--train-labels`
- `--valid-labels`
- `--train-scores`
- `--valid-scores`
- `--output`
- `--model-output`
- `--run-name`
- `--allow-partial`
- `--c`: Logistic Regression inverse regularization
- `--max-iter`

### `scripts/evaluate_log_loss.py`

Evaluates prediction probabilities.

Key args:

- `--labels`
- `--predictions`
- `--allow-partial`
- `--normalize`
- `--clip`

## Recommended From-Scratch Workflow

```bash
conda activate deepl

RUN_NAME=$(date +"%Y-%m-%d_%H-%M")

python scripts/create_validation_split.py

python scripts/score_gemma_rm.py \
  --input data/valid_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --load-in-4bit \
  --dtype float16 \
  --gpu-memory 7.6GiB \
  --save-every 100 \
  --resume

python scripts/score_gemma_rm.py \
  --input data/train_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --load-in-4bit \
  --dtype float16 \
  --gpu-memory 7.6GiB \
  --save-every 100 \
  --resume

python scripts/train_rm_calibrator.py \
  --run-name $RUN_NAME

python scripts/evaluate_log_loss.py \
  --predictions output/$RUN_NAME/RM_LogisticRegression/rm_calibrated_valid_predictions.csv
```

For a faster first run, score only 1000 train rows:

```bash
python scripts/score_gemma_rm.py \
  --input data/train_split.csv \
  --run-name $RUN_NAME \
  --max-length 512 \
  --batch-size 2 \
  --load-in-4bit \
  --dtype float16 \
  --gpu-memory 7.6GiB \
  --limit 1000 \
  --save-every 100 \
  --resume
```

Then train with explicit partial paths:

```bash
python scripts/train_rm_calibrator.py \
  --train-scores output/$RUN_NAME/RM_LogisticRegression/train_split_gemma_rm_scores_limit1000.csv \
  --valid-scores output/$RUN_NAME/RM_LogisticRegression/valid_split_gemma_rm_scores.csv \
  --output output/$RUN_NAME/RM_LogisticRegression/rm_calibrated_valid_predictions_partial.csv \
  --model-output output/$RUN_NAME/RM_LogisticRegression/rm_logistic_regression_model_limit1000.joblib \
  --allow-partial
```
