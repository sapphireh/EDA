"""Runtime, quantization, and LoRA helpers."""

from __future__ import annotations

import os
import random
from typing import Optional

import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def detect_attention_implementation() -> str:
    try:
        __import__("flash_attn")
        return "flash_attention_2"
    except Exception:
        return "eager"


def build_quantization_config(use_qlora: bool) -> Optional[BitsAndBytesConfig]:
    if not use_qlora:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_dtype(),
    )


def build_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> LoraConfig:
    """Build the same Llama LoRA configuration as the final experiment code."""

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    try:
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            use_rslora=False,
        )
    except TypeError:
        try:
            return LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
            )
        except TypeError:
            pass
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
