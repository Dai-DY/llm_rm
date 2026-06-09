from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    description: str
    defaults: dict[str, dict[str, Any]]


PROFILES = {
    "4090": HardwareProfile(
        name="4090",
        description="RTX 4090 24GB single GPU",
        defaults={
            "gemma_train": {
                "batch_size": 2,
                "eval_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "gemma_predict": {
                "batch_size": 2,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "qwen_train": {
                "batch_size": 4,
                "eval_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "qwen_predict": {
                "batch_size": 4,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "rm_score": {
                "batch_size": 2,
                "dtype": "bfloat16",
                "load_in_4bit": False,
                "gpu_memory": "23GiB",
                "cpu_memory": "48GiB",
            },
        },
    ),
    "h20": HardwareProfile(
        name="h20",
        description="Single NVIDIA H20 GPU",
        defaults={
            "gemma_train": {
                "batch_size": 8,
                "eval_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "gemma_predict": {
                "batch_size": 8,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "qwen_train": {
                "batch_size": 16,
                "eval_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "qwen_predict": {
                "batch_size": 16,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "rm_score": {
                "batch_size": 8,
                "dtype": "bfloat16",
                "load_in_4bit": False,
                "gpu_memory": "90GiB",
                "cpu_memory": "96GiB",
            },
        },
    ),
    "h20x2": HardwareProfile(
        name="h20x2",
        description="Two NVIDIA H20 GPUs with torchrun/DDP for training",
        defaults={
            "gemma_train": {
                "batch_size": 8,
                "eval_batch_size": 8,
                "gradient_accumulation_steps": 2,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "gemma_predict": {
                "batch_size": 8,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "qwen_train": {
                "batch_size": 16,
                "eval_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "qwen_predict": {
                "batch_size": 16,
                "dtype": "bfloat16",
                "load_in_4bit": False,
            },
            "rm_score": {
                "batch_size": 8,
                "dtype": "bfloat16",
                "load_in_4bit": False,
                "gpu_memory": "90GiB",
                "cpu_memory": "96GiB",
            },
        },
    ),
}


def detect_hardware_profile() -> str:
    try:
        import torch
    except ImportError:
        return "4090"

    if not torch.cuda.is_available():
        return "4090"

    device_count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index).lower() for index in range(device_count)]
    h20_count = sum("h20" in name for name in names)
    if h20_count >= 2:
        return "h20x2"
    if h20_count == 1 and device_count == 1:
        return "h20"
    return "4090"


def add_hardware_profile_argument(parser) -> None:
    parser.add_argument(
        "--hardware-profile",
        choices=["auto", *PROFILES.keys()],
        default="auto",
        help="Hardware defaults to use. auto detects 4090, single H20, or dual H20.",
    )


def resolve_hardware_profile(name: str) -> HardwareProfile:
    if name == "auto":
        name = detect_hardware_profile()
    return PROFILES[name]


def apply_profile_defaults(args, workload: str):
    profile = resolve_hardware_profile(args.hardware_profile)
    defaults = profile.defaults[workload]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.hardware_profile = profile.name
    args.hardware_description = profile.description
    return args


def local_rank() -> int | None:
    value = os.environ.get("LOCAL_RANK")
    return int(value) if value is not None else None


def configure_cuda_for_local_rank() -> int | None:
    rank = local_rank()
    if rank is None:
        return None

    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    return rank


def kbit_device_map():
    rank = configure_cuda_for_local_rank()
    if rank is None:
        return "auto"
    return {"": rank}


def max_memory_map(gpu_memory: str, cpu_memory: str):
    try:
        import torch
    except ImportError:
        return {"cpu": cpu_memory}

    if not torch.cuda.is_available():
        return {"cpu": cpu_memory}

    return {
        **{index: gpu_memory for index in range(torch.cuda.device_count())},
        "cpu": cpu_memory,
    }


def model_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device

    try:
        return next(model.parameters()).device
    except StopIteration:
        pass

    import torch

    rank = configure_cuda_for_local_rank()
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}" if rank is not None else "cuda")
    return torch.device("cpu")
