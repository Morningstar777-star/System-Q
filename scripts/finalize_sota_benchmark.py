"""
Finalize SOTA Benchmark Results with Optimal Null Calibration (gamma=1.25).
Produces full records, energy telemetry, latency distribution, and updates sota_benchmark_100_results.json.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
import numpy as np
from peft import PeftModel

from src.energy_monitor import EnergyContext
from src.model_loader import load_qwen_model_and_tokenizer
from src.system_q_scorer import SystemQDirectScorer


def finalize():
    print("=" * 80)
    print("      FINALIZING SOTA BENCHMARK EVALUATION (OPTIMAL GAMMA = 1.25)")
    print("=" * 80, flush=True)

    with open("data/benchmark_100.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("Loading base INT4 model and merging SOTA LoRA adapter...")
    tokenizer, base_model, _ = load_qwen_model_and_tokenizer(precision="4bit")
    adapter_path = "models/adapters/system_q_sota_direct_scorer"
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    scorer = SystemQDirectScorer(
        model=merged_model,
        tokenizer=tokenizer,
        temperature=1.0,
        null_calibration_weight=1.25,
        enable_null_calibration=True
    )

    print("Prewarming CUDA kernels and caching background priors...", flush=True)
    scorer.warmup()

    records = []
    latencies = []
    correct_scam = 0
    total_scam = 0
    correct_intent = 0
    total_intent = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)

    label_name = "System Q SOTA (Fused INT4 + Margin LoRA + De-Biasing)"

    with EnergyContext(label=label_name, num_decisions=len(questions)) as energy_ctx:
        for i, q in enumerate(questions):
            text = q["input"]
            options = q["options"]
            gt = q["ground_truth"]
            domain = q["domain"]

            res = scorer.predict(text, options)
            dt = res["latency_ms"]
            choice = res["decision"]
            latencies.append(dt)

            is_correct = (choice == gt)
            if domain == "scam_detection":
                total_scam += 1
                if is_correct:
                    correct_scam += 1
            else:
                total_intent += 1
                if is_correct:
                    correct_intent += 1

            records.append({
                "id": q["id"],
                "domain": domain,
                "ground_truth": gt,
                "prediction": choice,
                "is_correct": is_correct,
                "confidence": res["confidence"],
                "margin": res["margin"],
                "latency_ms": dt,
                "scores": res["scores"]
            })

            if (i + 1) % 25 == 0:
                print(f"  Processed {i+1}/{len(questions)} queries | Running Acc: {((correct_scam + correct_intent)/(i+1))*100:.1f}%", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    else:
        vram_peak = 0.0

    lat_arr = np.array(latencies)
    scam_acc = (correct_scam / max(1, total_scam)) * 100.0
    intent_acc = (correct_intent / max(1, total_intent)) * 100.0
    overall_acc = ((correct_scam + correct_intent) / len(questions)) * 100.0

    sota_summary = {
        "model": label_name,
        "num_questions": len(questions),
        "overall_accuracy": round(overall_acc, 2),
        "scam_detection_accuracy": round(scam_acc, 2),
        "intent_classification_accuracy": round(intent_acc, 2),
        "calibration_gamma": 1.25,
        "latency": {
            "median_ms": round(float(np.median(lat_arr)), 2),
            "mean_ms": round(float(np.mean(lat_arr)), 2),
            "p90_ms": round(float(np.percentile(lat_arr, 90)), 2),
            "p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
            "throughput_decisions_per_sec": round(1000.0 / float(np.mean(lat_arr)), 2)
        },
        "memory": {
            "peak_vram_gb": round(vram_peak, 3)
        },
        "energy": energy_ctx.results,
        "records": records
    }

    # Load baseline comparison from benchmark_100_results.json
    baseline_path = "results/raw/benchmark_100_results.json"
    with open(baseline_path, "r", encoding="utf-8") as f:
        previous_results = json.load(f)

    all_results = {
        "laya": previous_results["laya"],
        "system_q_zeroshot": previous_results["system_q_zeroshot"],
        "system_q_v2_finetuned": previous_results.get("system_q_finetuned"),
        "system_q_sota": sota_summary
    }

    out_json = "results/raw/sota_benchmark_100_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to: {out_json}", flush=True)

    print("\n" + "=" * 90)
    print("                    FINAL HEAD-TO-HEAD BENCHMARK: LAYA vs SYSTEM Q SOTA")
    print("=" * 90)
    print(f"{'Model':<46} | {'Overall':<9} | {'Scam':<8} | {'Intent':<8} | {'Median Lat':<11} | {'Peak VRAM'}")
    print("-" * 90)
    for k, s in all_results.items():
        if s is None:
            continue
        print(f"{s['model']:<46} | {s['overall_accuracy']:>7}% | {s['scam_detection_accuracy']:>6}% | {s['intent_classification_accuracy']:>6}% | {s['latency']['median_ms']:>8} ms | {s['memory']['peak_vram_gb']} GB")
    print("=" * 90, flush=True)


if __name__ == "__main__":
    finalize()
