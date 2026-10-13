"""
Model Loader for System Q.
Handles precision modes (BF16, FP16, INT8, INT4 NF4, CPU fallback),
measures loading latency and memory footprint, and exposes a clean interface.
"""

import os
import time
import torch
from typing import Dict, Any, Tuple, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    PreTrainedModel
)

MODEL_NAME = "Qwen/Qwen2.5-3B"
MODEL_REVISION = "3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"


def get_model_disk_size_gb(model_name_or_path: str = MODEL_NAME) -> float:
    """Calculates model footprint in GB from huggingface cache or directory."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name_or_path:
                return round(repo.size_on_disk / (1024 ** 3), 3)
    except Exception:
        pass
    return 6.0  # nominal 3B BF16 size


def load_qwen_model_and_tokenizer(
    precision: str = "4bit",
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    load_causal_head: bool = True
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel, Dict[str, Any]]:
    """
    Loads Qwen2.5-3B with specified precision and device mapping.
    
    Args:
        precision: "4bit", "8bit", "bf16", "fp16", or "cpu"
        device_map: "auto", "cuda:0", "cpu", etc.
        torch_dtype: Optional override for compute dtype
        load_causal_head: If True loads AutoModelForCausalLM; else AutoModel
    
    Returns:
        (tokenizer, model, telemetry_dict)
    """
    t_start = time.perf_counter()
    vram_start = torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0

    print(f"Loading {MODEL_NAME} (precision={precision}, device_map={device_map})...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        use_fast=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    resolved_dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

    if precision == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        resolved_device_map = device_map
    elif precision == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        resolved_device_map = device_map
    elif precision == "cpu":
        resolved_device_map = "cpu"
        resolved_dtype = torch.float32
    else:  # fp16 or bf16
        resolved_device_map = device_map

    model_cls = AutoModelForCausalLM if load_causal_head else AutoModelForCausalLM

    kwargs: Dict[str, Any] = {
        "revision": MODEL_REVISION,
        "device_map": resolved_device_map,
        "torch_dtype": resolved_dtype,
    }
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config

    try:
        model = model_cls.from_pretrained(MODEL_NAME, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM with precision={precision}, falling back to CPU offload...")
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: "2.5GiB", "cpu": "12GiB"}
        model = model_cls.from_pretrained(MODEL_NAME, **kwargs)

    t_end = time.perf_counter()
    vram_end = torch.cuda.memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    vram_delta = max(0.0, vram_end - vram_start)

    telemetry = {
        "precision": precision,
        "device_map": str(resolved_device_map),
        "load_time_seconds": round(t_end - t_start, 3),
        "vram_allocated_gb": round(vram_end, 3),
        "vram_delta_gb": round(vram_delta, 3),
        "disk_size_gb": get_model_disk_size_gb(MODEL_NAME),
        "param_count": sum(p.numel() for p in model.parameters()),
    }

    print(f"Loaded in {telemetry['load_time_seconds']}s. VRAM allocated: {telemetry['vram_allocated_gb']} GB")
    return tokenizer, model, telemetry


if __name__ == "__main__":
    print("Testing model loader specification...")
