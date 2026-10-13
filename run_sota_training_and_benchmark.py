"""
SOTA Training & Benchmark Runner:
1. Trains System Q on 140 rich scam & intent examples using Hybrid Contrastive Margin Loss.
2. Fuses/Merges LoRA weights directly into the base INT4 NF4 model (merge_and_unload).
3. Evaluates SOTA model on all 100 benchmark questions with warm prior caching and energy telemetry.
4. Generates comparison tables and updates benchmark records.
"""

import os
import gc
import json
import time
import torch
import numpy as np
from typing import Dict, Any, List

from src.hardware_detection import HardwareDetector
from src.energy_monitor import EnergyContext
from src.model_loader import load_qwen_model_and_tokenizer
from src.system_q_scorer import SystemQDirectScorer
from src.train_direct_scorer import train_direct_scorer_lora


def evaluate_system_q_sota(
    model,
    tokenizer,
    questions: List[Dict[str, Any]],
    label_name: str = "System Q SOTA (Fused INT4 + Margin LoRA)",
    enable_null_calib: bool = True
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"BENCHMARKING: {label_name.upper()}")
    print("=" * 70, flush=True)

    scorer = SystemQDirectScorer(
        model=model,
        tokenizer=tokenizer,
        temperature=1.0,
        enable_null_calibration=enable_null_calib
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

            if (i + 1) % 20 == 0:
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

    summary = {
        "model": label_name,
        "num_questions": len(questions),
        "overall_accuracy": round(overall_acc, 2),
        "scam_detection_accuracy": round(scam_acc, 2),
        "intent_classification_accuracy": round(intent_acc, 2),
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

    print(f"\n>>> RESULT FOR {label_name}:")
    print(f"  Overall Accuracy:      {summary['overall_accuracy']}%")
    print(f"  Scam Detection:        {summary['scam_detection_accuracy']}% ({correct_scam}/{total_scam})")
    print(f"  Intent Classification: {summary['intent_classification_accuracy']}% ({correct_intent}/{total_intent})")
    print(f"  Median Latency:        {summary['latency']['median_ms']} ms")
    print(f"  Mean Latency:          {summary['latency']['mean_ms']} ms")
    print(f"  P95 Latency:           {summary['latency']['p95_ms']} ms")
    print(f"  Throughput:            {summary['latency']['throughput_decisions_per_sec']} dec/s")
    print(f"  Peak VRAM:             {summary['memory']['peak_vram_gb']} GB")
    print(f"  Energy / Decision:     {energy_ctx.results.get('energy_mj_per_decision')} mJ", flush=True)

    return summary


def main():
    print("=" * 75)
    print("      SYSTEM Q: STATE-OF-THE-ART DECISION ENGINE TRAINING & BENCHMARK")
    print("=" * 75, flush=True)

    # 1. Load data
    with open("data/benchmark_100.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open("data/train_scam_intent_sota.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    print(f"Evaluation dataset: {len(questions)} test items (50 Scam, 50 Intent)")
    print(f"Training dataset:   {len(train_data)} items (70 Scam, 70 Intent, zero leakage)", flush=True)

    # 2. Load Base Model in INT4 NF4
    tokenizer, base_model, _ = load_qwen_model_and_tokenizer(precision="4bit")

    # 3. Train SOTA LoRA with Contrastive Margin Loss
    print("\n--- Initiating SOTA LoRA Training ---", flush=True)
    ft_model = train_direct_scorer_lora(
        base_model=base_model,
        tokenizer=tokenizer,
        training_data=train_data,
        output_adapter_dir="models/adapters/system_q_sota_direct_scorer",
        epochs=4,
        lr=2.5e-4,
        lora_r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        gradient_accumulation_steps=4,
        merge_weights_after_training=True
    )

    # Clean cache and trigger garbage collection
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Evaluate SOTA Model on all 100 benchmark questions
    sota_summary = evaluate_system_q_sota(
        model=ft_model,
        tokenizer=tokenizer,
        questions=questions,
        label_name="System Q SOTA (Fused INT4 + Margin LoRA)",
        enable_null_calib=True
    )

    # 5. Load baseline comparison from previous results
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

    # 6. Print Head-to-Head Comparison Table
    print("\n" + "=" * 85)
    print("                 HEAD-TO-HEAD BENCHMARK: LAYA vs SYSTEM Q SOTA")
    print("=" * 85)
    print(f"{'Model':<42} | {'Overall':<9} | {'Scam':<8} | {'Intent':<8} | {'Median Lat':<11} | {'Peak VRAM'}")
    print("-" * 85)
    for k, s in all_results.items():
        if s is None:
            continue
        print(f"{s['model']:<42} | {s['overall_accuracy']:>7}% | {s['scam_detection_accuracy']:>6}% | {s['intent_classification_accuracy']:>6}% | {s['latency']['median_ms']:>8} ms | {s['memory']['peak_vram_gb']} GB")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
