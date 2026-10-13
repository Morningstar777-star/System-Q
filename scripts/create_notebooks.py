"""
Helper to generate clean, reproducible Jupyter Notebooks for System Q.
Creates:
- 01_hardware_and_baseline.ipynb
- 02_decision_head.ipynb
- 03_finetuning.ipynb
- 04_optimization.ipynb
- 05_comparative_benchmark.ipynb
"""

import json
import os

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (mentalhealth)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.14"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True)
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True)
    }


# Notebook 1
nb1_cells = [
    md_cell("# System Q — 01: Hardware Detection and Baseline Inference\nThis notebook inspects system resources, verifies GPU VRAM on RTX 3050, and benchmarks Variant A (Original Qwen Baseline) under Constrained Generation and Direct Candidate Scoring."),
    code_cell("import torch\nfrom src.hardware_detection import HardwareDetector\ndetector = HardwareDetector()\ndetector.print_summary()"),
    code_cell("from src.model_loader import load_qwen_model_and_tokenizer\ntokenizer, model, meta = load_qwen_model_and_tokenizer(precision='4bit')\nprint(meta)"),
    code_cell("from src.baseline_inference import QwenConstrainedGenerationBaseline, QwenDirectCandidateScoringBaseline\ngen_base = QwenConstrainedGenerationBaseline(model, tokenizer)\nscore_base = QwenDirectCandidateScoringBaseline(model, tokenizer)\n\ntext = 'URGENT: Database connection pool exhausted.'\nopts = ['NON_URGENT', 'URGENT']\nprint('Generation:', gen_base.predict(text, opts))\nprint('Direct Score:', score_base.predict(text, opts))")
]

# Notebook 2
nb2_cells = [
    md_cell("# System Q — 02: Decision Head Architecture\nAttaches non-autoregressive classification heads to the pooled final hidden representations of Qwen2.5-3B and tests last-token pooling."),
    code_cell("from src.decision_model import SystemQDecisionModel\nsystem_q = SystemQDecisionModel(backbone=model, tokenizer=tokenizer, hidden_size=2048)\nsystem_q.freeze_backbone()"),
    code_cell("pred = system_q.predict('Duplicate charge on invoice #992', task_type='multiclass', options=['BILLING', 'TECH_SUPPORT', 'ACCOUNT', 'SALES'])\nprint('Decision:', pred)")
]

# Notebook 3
nb3_cells = [
    md_cell("# System Q — 03: Decision Head Training & LoRA Adaptation\nTrains decision heads on stratified training splits with teacher distillation guidance."),
    code_cell("from src.dataset_builder import build_and_save_datasets\nfrom src.training import SystemQTrainer\nsplits = build_and_save_datasets()\ntrainer = SystemQTrainer(system_q, tokenizer, splits['train'], splits['validation'], epochs=3)\nres = trainer.train_task_head(task_type='binary', label_list=['NON_URGENT', 'URGENT'])\nprint('Training Result:', res)")
]

# Notebook 4
nb4_cells = [
    md_cell("# System Q — 04: Quantization and Adaptive Early-Exit Inference\nEvaluates INT4 NF4 precision footprint and tests intermediate layer early-exit policy."),
    code_cell("from src.early_exit import EarlyExitDecisionModel\nee_model = EarlyExitDecisionModel(model, tokenizer, exit_layers=[12, 24, 36], policy='confidence', confidence_threshold=0.85)\nres = ee_model.predict('Production service down immediately', task_type='binary')\nprint('Early Exit Result:', res)")
]

# Notebook 5
nb5_cells = [
    md_cell("# System Q — 05: Comparative Benchmark & Pareto Analysis\nExecutes comparative evaluation across System Q, Laya, and Jev on the untouched test split and generates Pareto trade-offs."),
    code_cell("from src.report_generator import load_all_benchmark_results, generate_pareto_analysis, generate_plots\nresults = load_all_benchmark_results('results/raw')\npareto = generate_pareto_analysis(results)\nprint('Pareto Winners:', json.dumps(pareto, indent=2))"),
    code_cell("generate_plots(results, 'results/plots')\nprint('Plots generated in results/plots')")
]

notebooks = {
    "01_hardware_and_baseline.ipynb": nb1_cells,
    "02_decision_head.ipynb": nb2_cells,
    "03_finetuning.ipynb": nb3_cells,
    "04_optimization.ipynb": nb4_cells,
    "05_comparative_benchmark.ipynb": nb5_cells
}

for name, cells in notebooks.items():
    p = os.path.join(NOTEBOOKS_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(make_notebook(cells), f, indent=2)
    print(f"Created notebook: {p}")
