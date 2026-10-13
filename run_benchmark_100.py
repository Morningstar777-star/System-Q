"""
Benchmark Runner: Testing Laya and System Q Direct Scorer on 100 Questions.
Covers:
- 50 Scam Mail Detection Questions (SCAM vs LEGITIMATE)
- 50 Email Intent Classification Questions (BILLING, TECHNICAL_SUPPORT, ACCOUNT_ACCESS, SALES_INQUIRY, ACCOUNT_CANCELLATION)

Evaluates:
1. Laya (convaiinnovations/laya, ModernBERT-large RL agent)
2. System Q Direct Scorer (Pretrained Qwen2.5-3B INT4)
3. System Q Direct Scorer (Fine-Tuned with Candidate Margin LoRA + Null Calibration)
"""

import os
import json
import time
import psutil
import torch
import numpy as np
from typing import Dict, Any, List
import laya

from src.hardware_detection import HardwareDetector
from src.energy_monitor import EnergyContext
from src.model_loader import load_qwen_model_and_tokenizer
from src.system_q_scorer import SystemQDirectScorer
from src.train_direct_scorer import train_direct_scorer_lora


def run_laya_on_100(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluates Laya on all 100 benchmark questions."""
    print("\n" + "=" * 65)
    print("BENCHMARKING MODEL 1: LAYA (convaiinnovations/laya)")
    print("=" * 65)

    agent = laya.load("convaiinnovations/laya")
    device = agent.device if hasattr(agent, "device") else "cuda"
    print(f"Laya loaded successfully on {device}")

    # Warmup
    _ = agent.predict("Warmup query", {"intent": {"type": "choice", "instructions": "Select one", "criteria": ["A", "B"]}})

    records = []
    latencies = []
    correct_scam = 0
    total_scam = 0
    correct_intent = 0
    total_intent = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    with EnergyContext(label="Laya_100", num_decisions=len(questions)) as energy_ctx:
        for q in questions:
            text = q["input"]
            options = q["options"]
            gt = q["ground_truth"]
            domain = q["domain"]

            schema = {
                "decision": {
                    "type": "choice",
                    "instructions": f"Select the single best label matching: {options}",
                    "criteria": options
                }
            }

            t0 = time.perf_counter()
            try:
                res = agent.predict(text, schema)
                dt = (time.perf_counter() - t0) * 1000.0
                choice = res["answers"]["decision"].get("choice", options[0])
                probs = res["answers"]["decision"].get("probabilities", {})
            except Exception as e:
                dt = (time.perf_counter() - t0) * 1000.0
                choice = options[0]
                probs = {opt: 1.0 / len(options) for opt in options}

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
                "latency_ms": round(dt, 2),
                "scores": probs
            })

    vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    lat_arr = np.array(latencies)

    scam_acc = correct_scam / max(1, total_scam)
    intent_acc = correct_intent / max(1, total_intent)
    overall_acc = (correct_scam + correct_intent) / len(questions)

    summary = {
        "model": "Laya (ModernBERT-large RL Agent)",
        "num_questions": len(questions),
        "overall_accuracy": round(overall_acc * 100, 2),
        "scam_detection_accuracy": round(scam_acc * 100, 2),
        "intent_classification_accuracy": round(intent_acc * 100, 2),
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

    print(f"Laya Overall Accuracy: {summary['overall_accuracy']}%")
    print(f"  - Scam Detection:     {summary['scam_detection_accuracy']}% ({correct_scam}/{total_scam})")
    print(f"  - Intent Classification: {summary['intent_classification_accuracy']}% ({correct_intent}/{total_intent})")
    print(f"  - Median Latency:     {summary['latency']['median_ms']} ms (Throughput: {summary['latency']['throughput_decisions_per_sec']} dec/s)")
    print(f"  - Peak VRAM:          {summary['memory']['peak_vram_gb']} GB")
    print(f"  - Energy / Decision:  {energy_ctx.results.get('energy_mj_per_decision')} mJ")

    return summary


def run_system_q_on_100(
    model,
    tokenizer,
    questions: List[Dict[str, Any]],
    label_name: str = "System Q Direct Scorer",
    enable_null_calib: bool = True
) -> Dict[str, Any]:
    """Evaluates System Q Direct Scorer on all 100 benchmark questions."""
    print("\n" + "=" * 65)
    print(f"BENCHMARKING MODEL: {label_name.upper()}")
    print("=" * 65)

    scorer = SystemQDirectScorer(
        model=model,
        tokenizer=tokenizer,
        temperature=1.0,
        enable_null_calibration=enable_null_calib
    )

    # Warmup
    _ = scorer.predict("Warmup query", ["A", "B"])

    records = []
    latencies = []
    correct_scam = 0
    total_scam = 0
    correct_intent = 0
    total_intent = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)

    with EnergyContext(label=label_name, num_decisions=len(questions)) as energy_ctx:
        for q in questions:
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

    vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
    lat_arr = np.array(latencies)

    scam_acc = correct_scam / max(1, total_scam)
    intent_acc = correct_intent / max(1, total_intent)
    overall_acc = (correct_scam + correct_intent) / len(questions)

    summary = {
        "model": label_name,
        "num_questions": len(questions),
        "overall_accuracy": round(overall_acc * 100, 2),
        "scam_detection_accuracy": round(scam_acc * 100, 2),
        "intent_classification_accuracy": round(intent_acc * 100, 2),
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

    print(f"{label_name} Overall Accuracy: {summary['overall_accuracy']}%")
    print(f"  - Scam Detection:     {summary['scam_detection_accuracy']}% ({correct_scam}/{total_scam})")
    print(f"  - Intent Classification: {summary['intent_classification_accuracy']}% ({correct_intent}/{total_intent})")
    print(f"  - Median Latency:     {summary['latency']['median_ms']} ms (Throughput: {summary['latency']['throughput_decisions_per_sec']} dec/s)")
    print(f"  - Peak VRAM:          {summary['memory']['peak_vram_gb']} GB")
    print(f"  - Energy / Decision:  {energy_ctx.results.get('energy_mj_per_decision')} mJ")

    return summary


def main():
    print("=" * 70)
    print("   SYSTEM Q vs LAYA: 100-QUESTION HEAD-TO-HEAD BENCHMARK")
    print("=" * 70)

    # 1. Load benchmark questions
    with open("data/benchmark_100.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} evaluation questions.")

    # 2. Run Laya
    laya_summary = run_laya_on_100(questions)

    # 3. Load Qwen2.5-3B in INT4 NF4
    print("\nLoading Qwen2.5-3B Backbone (INT4 NF4)...")
    tokenizer, base_model, _ = load_qwen_model_and_tokenizer(precision="4bit")

    # 4. Run System Q (Pretrained Direct Scorer Baseline)
    pretrained_summary = run_system_q_on_100(
        base_model,
        tokenizer,
        questions,
        label_name="System Q Direct Scorer (Zero-Shot)",
        enable_null_calib=False
    )

    # 5. Run System Q with Innovation: Null-Context Prior De-biasing
    innovated_base_summary = run_system_q_on_100(
        base_model,
        tokenizer,
        questions,
        label_name="System Q Direct Scorer (+ Null-Calibration)",
        enable_null_calib=True
    )

    # 6. Fine-tune on specific tasks (Scam & Intent) using Candidate Logit Loss
    print("\n" + "=" * 65)
    print("TRAINING SYSTEM Q DIRECT SCORER ON SPECIFIC TASKS (LoRA)")
    print("=" * 65)
    with open("data/train_scam_intent.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    ft_model = train_direct_scorer_lora(
        base_model=base_model,
        tokenizer=tokenizer,
        training_data=train_data,
        output_adapter_dir="models/adapters/system_q_direct_scorer",
        epochs=3,
        lr=2e-4,
        lora_r=16,
        lora_alpha=32
    )

    # 7. Evaluate Fine-Tuned System Q Model on All 100 Questions
    finetuned_summary = run_system_q_on_100(
        ft_model,
        tokenizer,
        questions,
        label_name="System Q Direct Scorer (Fine-Tuned + Calibrated)",
        enable_null_calib=True
    )

    # 8. Save all results
    os.makedirs("results/raw", exist_ok=True)
    all_summaries = {
        "laya": laya_summary,
        "system_q_zeroshot": pretrained_summary,
        "system_q_calibrated": innovated_base_summary,
        "system_q_finetuned": finetuned_summary
    }

    with open("results/raw/benchmark_100_results.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    # 9. Print Final Head-to-Head Comparison Table
    print("\n" + "=" * 75)
    print("                  FINAL HEAD-TO-HEAD BENCHMARK SUMMARY (100 QUESTIONS)")
    print("=" * 75)
    print(f"{'Model':<40} | {'Overall Acc':<12} | {'Scam Acc':<10} | {'Intent Acc':<10} | {'Median Lat':<10} | {'VRAM'}")
    print("-" * 95)
    for k, s in all_summaries.items():
        print(f"{s['model']:<40} | {s['overall_accuracy']:>10}% | {s['scam_detection_accuracy']:>8}% | {s['intent_classification_accuracy']:>8}% | {s['latency']['median_ms']:>8} ms | {s['memory']['peak_vram_gb']} GB")
    print("=" * 75)


if __name__ == "__main__":
    main()
