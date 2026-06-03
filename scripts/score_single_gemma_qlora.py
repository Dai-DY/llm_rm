import argparse
import json

import _bootstrap  # noqa: F401
import torch

from Gemma2_QLoRA.data import build_compact_pair_text
from Gemma2_QLoRA.metrics import softmax
from Gemma2_QLoRA.predict import apply_adapter_config_defaults, load_model
from Gemma2_QLoRA.modeling import load_tokenizer
from hardware_profiles import (
    add_hardware_profile_argument,
    apply_profile_defaults,
    model_input_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score one response with a trained Gemma2 QLoRA preference classifier. "
            "The score is relative to a reference response, not an absolute reward."
        )
    )
    parser.add_argument("--model", default="models/sfairXC__FsfairX-Gemma2-RM-v0.1")
    parser.add_argument("--adapter", required=True, help="Trained LoRA adapter directory.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--response", required=True, help="Candidate response text.")
    parser.add_argument(
        "--reference-response",
        default="",
        help="Reference response to compare against. Defaults to an empty response.",
    )
    parser.add_argument("--max-length", type=int, default=1800)
    add_hardware_profile_argument(parser)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--disable-softcapping",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--classifier-head", choices=["linear", "mlp"], default=None)
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument("--head-hidden-ratio", type=float, default=None)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=None,
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return apply_profile_defaults(parser.parse_args(), "gemma_predict")


def main() -> None:
    args = apply_adapter_config_defaults(parse_args())
    tokenizer = load_tokenizer(args.adapter)
    model = load_model(args)

    text = build_compact_pair_text(
        json.dumps([args.prompt], ensure_ascii=False),
        json.dumps([args.response], ensure_ascii=False),
        json.dumps([args.reference_response], ensure_ascii=False),
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        padding=False,
    )
    device = model_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits.detach().float().cpu().numpy()[0]
    probs = softmax(logits.reshape(1, -1))[0]

    relative_reward_score = float(logits[0] - logits[1])
    probability_margin = float(probs[0] - probs[1])

    print(f"winner_model_a_probability={float(probs[0]):.8f}")
    print(f"winner_model_b_probability={float(probs[1]):.8f}")
    print(f"winner_tie_probability={float(probs[2]):.8f}")
    print(f"relative_reward_score_logit_margin={relative_reward_score:.8f}")
    print(f"relative_probability_margin={probability_margin:.8f}")


if __name__ == "__main__":
    main()
