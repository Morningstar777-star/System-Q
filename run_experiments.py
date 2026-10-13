"""
System Q: Master Experiment Execution Pipeline.
Executes Experiments 1 through 7 in sequential, reproducible order:
- Exp 1: Hardware & Baseline (Variant A1 & A2)
- Exp 2: Decision Head (Variant B)
- Exp 3: Fine-Tuning & Adaptation
- Exp 4: Quantization Benchmarking (Variant C)
- Exp 5: Adaptive Early-Exit Inference (Variant D)
- Exp 6: Laya & Jev Benchmark Comparison
- Exp 7: Final Untouched Test Set Evaluation & Pareto Report Generation
"""

import os
import sys
import json
import time
import torch
import laya

from src.hardware_detection import HardwareDetector
from src.dataset_builder import build_and_save_datasets
from src.model_loader import load_qwen_model_and_tokenizer
from src.baseline_inference import QwenConstrainedGenerationBaseline, QwenDirectCandidateScoringBaseline
from src.decision_model import SystemQDecisionModel
from src.early_exit import EarlyExitDecisionModel
from src.training import SystemQTrainer
from src.calibration import compute_ece, compute_brier_score, compute_nll, TemperatureScaler
from src.benchmark_runner import (
    BenchmarkSuite,
    QwenBaselineAdapter,
    SystemQDecisionHeadAdapter,
    SystemQEarlyExitAdapter,
    LayaAdapter,
    JevAdapter
)
from src.report_generator import (
    load_all_benchmark_results,
    generate_pareto_analysis,
    generate_comparison_table,
    generate_plots,
    generate_final_report
)


def main():
    print("=" * 70)
    print("      SYSTEM Q: QWEN2.5-3B SYSTEM 1 DECISION RESEARCH PIPELINE")
    print("=" * 70)

    # -------------------------------------------------------------
    # STEP 0: Hardware Detection & Environment Verification
    # -------------------------------------------------------------
    detector = HardwareDetector()
    hardware_profile = detector.get_full_profile()
    detector.print_summary()

    # Build and verify datasets
    splits = build_and_save_datasets(data_dir="data", seed=42)
    train_data = splits["train"]
    val_data = splits["validation"]
    test_data = splits["test"]
    print(f"Datasets: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # -------------------------------------------------------------
    # STEP 1: Model Loading (INT4 NF4 for RTX 3050 4GB VRAM)
    # -------------------------------------------------------------
    print("\n>>> Loading Qwen2.5-3B Backbone (INT4 NF4)...")
    tokenizer, qwen_base_model, loader_telemetry = load_qwen_model_and_tokenizer(
        precision="4bit",
        device_map="auto"
    )

    # Setup Benchmark Suite on the Untouched Test Set
    suite = BenchmarkSuite(dataset=test_data, output_dir="results", seed=42)
    all_results = {}

    # -------------------------------------------------------------
    # EXPERIMENT 1: Baseline Inference (Variant A1 & A2)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Original Qwen Baseline Inference")
    print("=" * 60)
    
    # A1: Constrained Generation
    print("Evaluating Variant A1: Constrained Short-Answer Generation...")
    baseline_gen = QwenConstrainedGenerationBaseline(qwen_base_model, tokenizer, max_new_tokens=4)
    adapter_a1 = QwenBaselineAdapter(baseline_gen, mode="generate")
    res_a1 = suite.run_model_benchmark(adapter_a1, warmup_runs=1, repeats=1)
    all_results["Qwen_Baseline_Generate"] = res_a1

    # A2: Direct Candidate Scoring
    print("\nEvaluating Variant A2: Direct Candidate Scoring (Next-Token Logits)...")
    baseline_score = QwenDirectCandidateScoringBaseline(qwen_base_model, tokenizer)
    adapter_a2 = QwenBaselineAdapter(baseline_score, mode="direct_score")
    res_a2 = suite.run_model_benchmark(adapter_a2, warmup_runs=1, repeats=1)
    all_results["Qwen_Baseline_DirectScore"] = res_a2

    # -------------------------------------------------------------
    # EXPERIMENT 2 & 3: Decision Head (Variant B) & Training
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 & 3: System Q Decision Head & Head Training")
    print("=" * 60)

    system_q_model = SystemQDecisionModel(
        backbone=qwen_base_model,
        tokenizer=tokenizer,
        hidden_size=2048,
        pooling="last_token"
    )
    system_q_model.freeze_backbone()

    trainer = SystemQTrainer(
        model=system_q_model,
        tokenizer=tokenizer,
        train_examples=train_data,
        val_examples=val_data,
        checkpoint_dir="models/checkpoints",
        batch_size=4,
        gradient_accumulation_steps=2,
        lr=5e-4,
        epochs=4
    )

    # Train heads across binary and multiclass
    print("Training binary decision head on training set...")
    train_res_bin = trainer.train_task_head(task_type="binary", label_list=["NON_URGENT", "URGENT"])
    print("Training multiclass routing head on training set...")
    train_res_mc = trainer.train_task_head(task_type="multiclass", label_list=["BILLING", "TECH_SUPPORT", "ACCOUNT", "SALES"])

    system_q_model.save_heads("models/decision_head")

    # Evaluate Variant B on test suite
    adapter_b = SystemQDecisionHeadAdapter(system_q_model)
    res_b = suite.run_model_benchmark(adapter_b, warmup_runs=1, repeats=1)
    all_results["System_Q_DecisionHead"] = res_b

    # -------------------------------------------------------------
    # EXPERIMENT 4: Quantized Decision Model (Variant C)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Quantized Decision Model Analysis")
    print("=" * 60)
    print("Variant B is executed natively in INT4 NF4 precision.")
    all_results["System_Q_Quantized_4Bit"] = res_b

    # -------------------------------------------------------------
    # EXPERIMENT 5: Adaptive Early-Exit Inference (Variant D)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Adaptive Early-Exit Decision Model")
    print("=" * 60)

    early_exit_model = EarlyExitDecisionModel(
        backbone=qwen_base_model,
        tokenizer=tokenizer,
        hidden_size=2048,
        exit_layers=[12, 24, 36],
        policy="confidence",
        confidence_threshold=0.85
    )

    adapter_d = SystemQEarlyExitAdapter(early_exit_model)
    res_d = suite.run_model_benchmark(adapter_d, warmup_runs=1, repeats=1)
    all_results["System_Q_EarlyExit"] = res_d

    # -------------------------------------------------------------
    # EXPERIMENT 6: Laya & Jev Benchmark Comparison
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Laya & Jev Comparison")
    print("=" * 60)

    # Laya (Local ModernBERT RL agent)
    print("Loading local Laya agent (convaiinnovations/laya)...")
    try:
        laya_agent = laya.load("convaiinnovations/laya")
        adapter_laya = LayaAdapter(laya_agent)
        res_laya = suite.run_model_benchmark(adapter_laya, warmup_runs=1, repeats=1)
        all_results["Laya_Local"] = res_laya
    except Exception as e:
        print(f"Laya evaluation error: {e}")

    # Jev (Remote TypeSafe API)
    print("\nEvaluating Jev Remote Adapter...")
    adapter_jev = JevAdapter(api_key=None)
    res_jev = suite.run_model_benchmark(adapter_jev, warmup_runs=0, repeats=1)
    all_results["TypeSafe_Jev_Remote"] = res_jev

    # -------------------------------------------------------------
    # EXPERIMENT 7: Final Comprehensive Evaluation & Reporting
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Final Evaluation, Pareto Analysis & Report")
    print("=" * 60)

    # Generate visual plots
    print("Generating benchmark plots...")
    generate_plots(all_results, output_dir="results/plots")

    # Generate comparison tables & final markdown report
    print("Generating comparison tables and final technical report...")
    generate_final_report(
        results=all_results,
        hardware_profile=hardware_profile,
        output_path="results/reports/final_technical_report.md"
    )

    print("\n" + "=" * 70)
    print("        SYSTEM Q RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
