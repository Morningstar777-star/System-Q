"""
Evaluate System Q SOTA adapter with and without null calibration
to measure the exact impact of calibration on the fine-tuned logits.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
import numpy as np
from peft import PeftModel

from src.model_loader import load_qwen_model_and_tokenizer
from src.system_q_scorer import SystemQDirectScorer


def test_variations():
    with open("data/benchmark_100.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    print("Loading base INT4 model...")
    tokenizer, base_model, _ = load_qwen_model_and_tokenizer(precision="4bit")

    print("Attaching SOTA adapter...")
    adapter_path = "models/adapters/system_q_sota_direct_scorer"
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Fusing LoRA weights into base model (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    configs = [
        {"name": "No Calibration (Raw Fine-Tuned Logits)", "enable_null": False, "gamma": 0.0, "temp": 1.0},
        {"name": "Null Calib gamma=0.10", "enable_null": True, "gamma": 0.10, "temp": 1.0},
        {"name": "Null Calib gamma=0.20 (Current)", "enable_null": True, "gamma": 0.20, "temp": 1.0},
        {"name": "Temperature=0.5 (Sharper decisions)", "enable_null": False, "gamma": 0.0, "temp": 0.5},
    ]

    results_summary = []

    for cfg in configs:
        scorer = SystemQDirectScorer(
            model=merged_model,
            tokenizer=tokenizer,
            temperature=cfg["temp"],
            null_calibration_weight=cfg["gamma"],
            enable_null_calibration=cfg["enable_null"]
        )
        scorer.warmup()

        correct_scam = 0
        total_scam = 0
        correct_intent = 0
        total_intent = 0
        lats = []

        for q in questions:
            t0 = time.perf_counter()
            res = scorer.predict(q["input"], q["options"])
            dt = (time.perf_counter() - t0) * 1000.0
            lats.append(dt)

            gt = q["ground_truth"]
            is_correct = (res["decision"] == gt)
            if q["domain"] == "scam_detection":
                total_scam += 1
                if is_correct:
                    correct_scam += 1
            else:
                total_intent += 1
                if is_correct:
                    correct_intent += 1

        scam_acc = (correct_scam / total_scam) * 100.0
        intent_acc = (correct_intent / total_intent) * 100.0
        overall_acc = ((correct_scam + correct_intent) / len(questions)) * 100.0
        med_lat = np.median(lats)

        print(f"\n[{cfg['name']}]")
        print(f"  Overall Acc: {overall_acc:.1f}% | Scam: {scam_acc:.1f}% ({correct_scam}/{total_scam}) | Intent: {intent_acc:.1f}% ({correct_intent}/{total_intent}) | Lat: {med_lat:.1f} ms")

        results_summary.append({
            "config": cfg,
            "overall_accuracy": overall_acc,
            "scam_accuracy": scam_acc,
            "intent_accuracy": intent_acc,
            "median_lat_ms": round(float(med_lat), 2)
        })

    with open("results/raw/sota_calibration_ablation.json", "w") as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    test_variations()
