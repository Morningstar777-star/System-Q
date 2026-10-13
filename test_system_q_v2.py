"""
Fast evaluation and fine-tuning runner for System Q 2.0 (Anchor-Score Architecture).
Evaluates zero-shot performance and fine-tuned performance on the 100-question benchmark.
"""

import os
import sys
import json
import time
import torch
import numpy as np

from src.model_loader import load_qwen_model_and_tokenizer
from src.system_q_scorer import SystemQDirectScorer
from src.train_direct_scorer import train_direct_scorer_lora


def evaluate_scorer(scorer, questions, label="Model"):
    print(f"\n--- Evaluating {label} on {len(questions)} Questions ---", flush=True)
    correct_scam = 0
    total_scam = 0
    correct_intent = 0
    total_intent = 0
    latencies = []
    records = []

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
            "latency_ms": dt
        })

    lat_arr = np.array(latencies)
    scam_acc = (correct_scam / max(1, total_scam)) * 100.0
    intent_acc = (correct_intent / max(1, total_intent)) * 100.0
    overall_acc = ((correct_scam + correct_intent) / len(questions)) * 100.0

    print(f"[{label}] Overall: {overall_acc:.1f}% | Scam: {scam_acc:.1f}% ({correct_scam}/{total_scam}) | Intent: {intent_acc:.1f}% ({correct_intent}/{total_intent})", flush=True)
    print(f"[{label}] Latency Median: {np.median(lat_arr):.2f} ms | Mean: {np.mean(lat_arr):.2f} ms | P95: {np.percentile(lat_arr, 95):.2f} ms", flush=True)

    return {
        "label": label,
        "overall_accuracy": round(overall_acc, 2),
        "scam_accuracy": round(scam_acc, 2),
        "intent_accuracy": round(intent_acc, 2),
        "latency_median_ms": round(float(np.median(lat_arr)), 2),
        "records": records
    }


def main():
    print("Loading 100 evaluation questions and 35 training examples...", flush=True)
    with open("data/benchmark_100.json", "r", encoding="utf-8") as f:
        test_questions = json.load(f)
    with open("data/train_scam_intent.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    # Load Qwen2.5-3B in INT4 NF4
    tokenizer, base_model, _ = load_qwen_model_and_tokenizer(precision="4bit")

    # 1. Zero-Shot Anchor Scorer
    scorer_v2 = SystemQDirectScorer(
        model=base_model,
        tokenizer=tokenizer,
        temperature=1.0,
        enable_null_calibration=True
    )
    res_zeroshot = evaluate_scorer(scorer_v2, test_questions, label="System Q 2.0 (Zero-Shot)")

    # 2. Train LoRA with Candidate Logit Loss
    print("\nTraining LoRA on Scam & Intent tasks...", flush=True)
    ft_model = train_direct_scorer_lora(
        base_model=base_model,
        tokenizer=tokenizer,
        training_data=train_data,
        output_adapter_dir="models/adapters/system_q_v2_direct_scorer",
        epochs=3,
        lr=2e-4,
        lora_r=16,
        lora_alpha=32
    )

    # 3. Fine-Tuned Anchor Scorer
    scorer_ft = SystemQDirectScorer(
        model=ft_model,
        tokenizer=tokenizer,
        temperature=1.0,
        enable_null_calibration=True
    )
    res_finetuned = evaluate_scorer(scorer_ft, test_questions, label="System Q 2.0 (Fine-Tuned)")

    # Save summary
    results_path = "results/raw/system_q_v2_comparison.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "zero_shot": res_zeroshot,
            "fine_tuned": res_finetuned
        }, f, indent=2)
    print(f"\nResults saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
