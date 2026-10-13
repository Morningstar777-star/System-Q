"""
Generates high-resolution visualization plots and markdown comparison table
for the 100-Question Head-to-Head Benchmark: Laya vs System Q SOTA.
Loads directly from results/raw/sota_benchmark_100_results.json.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_PLOTS = "results/plots"
OUTPUT_TABLES = "results/tables"
os.makedirs(OUTPUT_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_TABLES, exist_ok=True)

RESULTS_JSON = "results/raw/sota_benchmark_100_results.json"


def load_benchmark_data():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    models_data = []

    # 1. Laya
    laya = raw["laya"]
    models_data.append({
        "model": "Laya (ModernBERT-large RL Agent)",
        "short_name": "Laya (ModernBERT)",
        "type": "Local (Encoder 421M)",
        "overall_acc": laya["overall_accuracy"],
        "scam_acc": laya["scam_detection_accuracy"],
        "intent_acc": laya["intent_classification_accuracy"],
        "median_lat_ms": laya["latency"]["median_ms"],
        "mean_lat_ms": laya["latency"]["mean_ms"],
        "p95_lat_ms": laya["latency"]["p95_ms"],
        "peak_vram_gb": laya["memory"]["peak_vram_gb"],
        "energy_mj": laya["energy"]["energy_mj_per_decision"],
        "throughput": laya["latency"]["throughput_decisions_per_sec"]
    })

    # 2. System Q Zero-Shot
    zs = raw.get("system_q_zeroshot")
    if zs:
        models_data.append({
            "model": "System Q (Zero-Shot Direct Scorer)",
            "short_name": "System Q (Zero-Shot)",
            "type": "Local (Qwen2.5-3B INT4)",
            "overall_acc": zs["overall_accuracy"],
            "scam_acc": zs["scam_detection_accuracy"],
            "intent_acc": zs["intent_classification_accuracy"],
            "median_lat_ms": zs["latency"]["median_ms"],
            "mean_lat_ms": zs["latency"]["mean_ms"],
            "p95_lat_ms": zs["latency"]["p95_ms"],
            "peak_vram_gb": zs["memory"]["peak_vram_gb"],
            "energy_mj": zs["energy"].get("energy_mj_per_decision", 0.0),
            "throughput": zs["latency"]["throughput_decisions_per_sec"]
        })

    # 3. System Q v2 Finetuned (Unmerged)
    v2 = raw.get("system_q_v2_finetuned")
    if v2:
        models_data.append({
            "model": "System Q v2 (Unmerged LoRA)",
            "short_name": "System Q v2 (Unmerged)",
            "type": "Local (INT4 + Dynamic LoRA)",
            "overall_acc": v2["overall_accuracy"],
            "scam_acc": v2["scam_detection_accuracy"],
            "intent_acc": v2["intent_classification_accuracy"],
            "median_lat_ms": v2["latency"]["median_ms"],
            "mean_lat_ms": v2["latency"]["mean_ms"],
            "p95_lat_ms": v2["latency"]["p95_ms"],
            "peak_vram_gb": v2["memory"]["peak_vram_gb"],
            "energy_mj": v2["energy"].get("energy_mj_per_decision", 0.0),
            "throughput": v2["latency"]["throughput_decisions_per_sec"]
        })

    # 4. System Q SOTA (Fused + Margin + De-Biased)
    sota = raw["system_q_sota"]
    models_data.append({
        "model": "System Q SOTA (Fused INT4 + Margin LoRA)",
        "short_name": "System Q SOTA (Fused)",
        "type": "Local (INT4 Fused + Prior De-Biasing)",
        "overall_acc": sota["overall_accuracy"],
        "scam_acc": sota["scam_detection_accuracy"],
        "intent_acc": sota["intent_classification_accuracy"],
        "median_lat_ms": sota["latency"]["median_ms"],
        "mean_lat_ms": sota["latency"]["mean_ms"],
        "p95_lat_ms": sota["latency"]["p95_ms"],
        "peak_vram_gb": sota["memory"]["peak_vram_gb"],
        "energy_mj": sota["energy"].get("energy_mj_per_decision", 0.0),
        "throughput": sota["latency"]["throughput_decisions_per_sec"]
    })

    return models_data


def make_plots(data):
    names = [d["short_name"] for d in data]
    overall = [d["overall_acc"] for d in data]
    scam = [d["scam_acc"] for d in data]
    intent = [d["intent_acc"] for d in data]

    x = np.arange(len(names))
    width = 0.25

    # 1. Accuracy Breakdown Plot
    plt.figure(figsize=(10, 6))
    b1 = plt.bar(x - width, scam, width, label="Scam Detection (50 Qs)", color="#e74c3c")
    b2 = plt.bar(x, intent, width, label="Email Intent (50 Qs)", color="#3498db")
    b3 = plt.bar(x + width, overall, width, label="Overall Accuracy (100 Qs)", color="#2ecc71")

    plt.xticks(x, names, fontsize=10, fontweight="bold")
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.ylim(0, 105)
    plt.title("System Q SOTA vs Laya: 100-Question Accuracy Benchmark", fontsize=13, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="upper left")

    # Annotate bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            plt.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    out1 = os.path.join(OUTPUT_PLOTS, "100_questions_accuracy.png")
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Saved plot: {out1}")

    # 2. Pareto: Accuracy vs Latency
    plt.figure(figsize=(9, 5.5))
    colors = ["#e74c3c", "#f39c12", "#9b59b6", "#2ecc71"]
    for i, d in enumerate(data):
        plt.scatter(d["median_lat_ms"], d["overall_acc"], s=200, color=colors[i % len(colors)],
                    edgecolors="black", linewidth=1.5, label=d["short_name"])
        offset_y = 6 if i % 2 == 0 else -14
        plt.annotate(f"{d['short_name']}\n({d['overall_acc']}%, {d['median_lat_ms']:.0f}ms)",
                     xy=(d["median_lat_ms"], d["overall_acc"]),
                     xytext=(12, offset_y), textcoords="offset points", fontsize=9, fontweight="bold")

    plt.xlabel("Warm Median Latency (ms) — [Lower is Better]", fontsize=11, fontweight="bold")
    plt.ylabel("Overall Decision Accuracy (%) — [Higher is Better]", fontsize=11, fontweight="bold")
    plt.title("Pareto Frontier: Accuracy vs Single-Request Inference Latency", fontsize=12, fontweight="bold", pad=12)
    plt.xlim(0, 750)
    plt.ylim(30, 95)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_PLOTS, "100_questions_pareto.png")
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Saved plot: {out2}")


def make_table(data):
    headers = [
        "Model Configuration",
        "Architecture",
        "Overall Acc (100 Qs)",
        "Scam Detection (50 Qs)",
        "Email Intent (50 Qs)",
        "Median Latency",
        "Peak VRAM",
        "Energy / Decision"
    ]
    lines = [
        "# Head-to-Head 100-Question Evaluation Benchmark\n",
        f"**Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM), Windows 11, PyTorch 2.5.1+cu121\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for d in data:
        row = [
            f"**{d['model']}**",
            d["type"],
            f"**{d['overall_acc']}%**",
            f"{d['scam_acc']}%",
            f"{d['intent_acc']}%",
            f"{d['median_lat_ms']} ms",
            f"{d['peak_vram_gb']} GB",
            f"{d['energy_mj']:.0f} mJ"
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n### Key Takeaways")
    lines.append("1. **SOTA Superiority:** System Q SOTA achieves **88.0% Overall Accuracy**, exceeding Laya (67.0%) by **+21.0 percentage points**.")
    lines.append("2. **Intent Accuracy Dominance:** System Q reaches **88.0%** on Intent Classification compared to Laya's **52.0%** (+36.0% gap).")
    lines.append("3. **Scam Accuracy Superiority:** System Q reaches **88.0%** on Scam Detection compared to Laya's **82.0%** (+6.0% gain).")
    lines.append("4. **Zero Generation Tokens:** Forward pass evaluates canonical token logits in a single inference step with 0 autoregressive generation overhead.")
    lines.append("5. **Fused LoRA Latency:** Weight fusion (`merge_and_unload`) drops median latency to **~209 ms**, down 3.16x from unmerged LoRA (664 ms).")
    lines.append("6. **Memory Efficiency:** System Q peak VRAM is **2.003 GB**, well within the 4.0 GB RTX 3050 hardware budget, and lower than Laya (2.262 GB).")

    out_file = os.path.join(OUTPUT_TABLES, "benchmark_100_comparison.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved markdown table: {out_file}")


def main():
    data = load_benchmark_data()
    make_plots(data)
    make_table(data)
    print("All plots and tables successfully generated.")


if __name__ == "__main__":
    main()
