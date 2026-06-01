from pathlib import Path

import pandas as pd

from RM_LogisticRegression.data import build_reward_text, text_char_length


def progress_iter(iterable, total: int, initial: int, description: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(
        iterable,
        total=total,
        initial=initial,
        desc=description,
        unit="row",
        dynamic_ncols=True,
    )


def load_reward_model(
    model_path: str,
    load_in_4bit: bool,
    dtype_name: str,
    gpu_memory: str,
    cpu_memory: str,
):
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    dtype_by_name = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype_by_name[dtype_name],
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "quantization_config": quantization_config,
        "device_map": "auto",
        "max_memory": {0: gpu_memory, "cpu": cpu_memory},
    }
    if dtype_name != "auto":
        model_kwargs["torch_dtype"] = dtype_by_name[dtype_name]

    model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
    model.config.use_cache = False
    model.eval()
    return tokenizer, model


def score_texts(
    tokenizer,
    model,
    texts: list[str],
    max_length: int,
    batch_size: int,
) -> list[float]:
    import torch

    scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        batch_scores = logits.detach().float().view(logits.shape[0], -1)[:, 0].cpu()
        scores.extend(float(score) for score in batch_scores.tolist())
    return scores


def write_scores(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def score_dataframe(
    df: pd.DataFrame,
    tokenizer,
    model,
    output_path: Path,
    max_length: int,
    batch_size: int,
    save_every: int,
    existing_rows: list[dict] | None = None,
) -> None:
    rows = existing_rows or []
    scored_ids = {row["id"] for row in rows}
    total_rows = len(df)
    remaining_df = df[~df["id"].isin(scored_ids)]
    remaining_rows = len(remaining_df)

    print(
        "[score] Starting RM scoring: "
        f"total_rows={total_rows}, already_scored={total_rows - remaining_rows}, "
        f"remaining={remaining_rows}"
    )

    iterator = progress_iter(
        remaining_df.iterrows(),
        total=remaining_rows,
        initial=0,
        description="[score] rows",
    )
    for _, row in iterator:
        text_a = build_reward_text(row["prompt"], row["response_a"])
        text_b = build_reward_text(row["prompt"], row["response_b"])
        score_a, score_b = score_texts(
            tokenizer,
            model,
            [text_a, text_b],
            max_length=max_length,
            batch_size=batch_size,
        )
        prompt_len = text_char_length(row["prompt"])
        response_a_len = text_char_length(row["response_a"])
        response_b_len = text_char_length(row["response_b"])
        rows.append(
            {
                "id": row["id"],
                "score_a": score_a,
                "score_b": score_b,
                "score_diff": score_a - score_b,
                "score_abs_diff": abs(score_a - score_b),
                "prompt_len": prompt_len,
                "response_a_len": response_a_len,
                "response_b_len": response_b_len,
                "response_len_diff": response_a_len - response_b_len,
            }
        )

        completed = len(rows)
        if completed % save_every == 0:
            write_scores(rows, output_path)
            print(f"[score] checkpoint: wrote {completed} rows to {output_path}")

    write_scores(rows, output_path)
    print(f"[score] Done. Wrote {output_path} with {len(rows)} rows")
