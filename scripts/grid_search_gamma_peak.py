"""
Find the global optimum of null-context prior de-biasing (gammas 0.85 to 1.10).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from peft import PeftModel

from src.model_loader import load_qwen_model_and_tokenizer
from src.system_q_scorer import SystemQDirectScorer


def run_grid_peak():
    with open("data/benchmark_100.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    tokenizer, base_model, _ = load_qwen_model_and_tokenizer(precision="4bit")
    adapter_path = "models/adapters/system_q_sota_direct_scorer"
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    gammas = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    best_acc = 0.0
    best_gamma = 0.80

    print(f"{'Gamma':<8} | {'Overall':<9} | {'Scam':<8} | {'Intent':<8}")
    print("-" * 42, flush=True)

    for g in gammas:
        scorer = SystemQDirectScorer(
            model=merged_model,
            tokenizer=tokenizer,
            temperature=1.0,
            null_calibration_weight=g,
            enable_null_calibration=True
        )
        scorer.warmup()

        correct_scam = 0
        total_scam = 0
        correct_intent = 0
        total_intent = 0

        for q in questions:
            res = scorer.predict(q["input"], q["options"])
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
        overall = ((correct_scam + correct_intent) / len(questions)) * 100.0
        print(f"{g:<8.2f} | {overall:>7.1f}% | {scam_acc:>6.1f}% | {intent_acc:>6.1f}%", flush=True)

        if overall > best_acc:
            best_acc = overall
            best_gamma = g

    print(f"\nGlobal Peak Gamma: {best_gamma} with Overall Accuracy: {best_acc:.1f}%", flush=True)


if __name__ == "__main__":
    run_grid_peak()
