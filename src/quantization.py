"""
Quantization Analysis and Execution for System Q (Variant C).
Evaluates FP16/BF16, INT8, and INT4 NF4 precision formats on the RTX 3050.
Measures memory, latency, and representation consistency.
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, BitsAndBytesConfig

from src.model_loader import load_qwen_model_and_tokenizer, MODEL_NAME
from src.decision_heads import pool_hidden_states


def compare_quantization_modes(
    test_inputs: List[str],
    precision_modes: List[str] = ["4bit", "8bit", "fp16"]
) -> Dict[str, Any]:
    """
    Evaluates each quantization mode, measuring VRAM footprint, warm latency,
    and cosine consistency of representations.
    """
    results = {}
    baseline_reps = None

    for mode in precision_modes:
        print(f"\n--- Evaluating Precision Mode: {mode} ---")
        try:
            tokenizer, model, loader_meta = load_qwen_model_and_tokenizer(precision=mode)
            device = next(model.parameters()).device

            # Warmup
            warm_prompt = "Warmup check for quantization runtime."
            inputs = tokenizer(warm_prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                _ = model(**inputs, output_hidden_states=True)

            latencies = []
            representations = []

            for text in test_inputs:
                inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                t0 = time.perf_counter()
                with torch.inference_mode():
                    out = model(**inp, output_hidden_states=True)
                    last_hidden = out.hidden_states[-1]
                    pooled = pool_hidden_states(last_hidden, inp["attention_mask"], pooling="last_token")
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)
                representations.append(pooled.detach().cpu())

            vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
            avg_lat = sum(latencies) / len(latencies)

            # Measure consistency against first evaluated mode
            all_reps = torch.cat(representations, dim=0)
            if baseline_reps is None:
                baseline_reps = all_reps
                cosine_sim = 1.0
            else:
                sims = F.cosine_similarity(all_reps.float(), baseline_reps.float(), dim=-1)
                cosine_sim = round(float(sims.mean().item()), 4)

            results[mode] = {
                "success": True,
                "load_time_s": loader_meta["load_time_seconds"],
                "vram_allocated_gb": loader_meta["vram_allocated_gb"],
                "vram_peak_gb": round(vram_peak, 3),
                "avg_latency_ms": round(avg_lat, 2),
                "cosine_similarity_to_base": cosine_sim,
                "error": None
            }

            # Cleanup VRAM before next mode
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Mode {mode} failed with error: {e}")
            results[mode] = {
                "success": False,
                "load_time_s": None,
                "vram_allocated_gb": None,
                "vram_peak_gb": None,
                "avg_latency_ms": None,
                "cosine_similarity_to_base": None,
                "error": str(e)
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    print("Testing quantization module...")
