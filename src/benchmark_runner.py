"""
Benchmark Runner and Normalized Model Adapters for System Q, Laya, and Jev.
Implements the unified DecisionModel interface, standardized output schema,
and comprehensive evaluation across Accuracy, Calibration, Latency, Memory, and Energy.
"""

import os
import time
import json
import random
import psutil
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from src.energy_monitor import EnergyContext
from src.calibration import compute_ece, compute_brier_score, compute_nll, evaluate_selective_risk


class DecisionModel:
    """Standardized decision model interface."""

    def __init__(self, name: str, is_remote: bool = False):
        self.name = name
        self.is_remote = is_remote

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes input_data dict:
        {
            "input": str,
            "task_type": str,
            "options": List[str]
        }
        
        Returns normalized schema:
        {
            "decision": str,
            "scores": Dict[str, float],
            "latency_ms": float,
            "error": Optional[str],
            "metadata": Dict[str, Any]
        }
        """
        raise NotImplementedError


class QwenBaselineAdapter(DecisionModel):
    """Adapter for Variant A: Original Qwen2.5-3B Causal Baseline."""

    def __init__(self, baseline_model, mode: str = "direct_score"):
        super().__init__(name=f"Qwen2.5-3B_{mode}", is_remote=False)
        self.baseline = baseline_model
        self.mode = mode

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data["input"]
        options = input_data["options"]
        try:
            res = self.baseline.predict(text, options)
            return {
                "decision": res["decision"],
                "scores": res["scores"],
                "latency_ms": res["latency_ms"],
                "error": None,
                "metadata": {
                    "mode": self.mode,
                    "generated_tokens": res.get("generated_tokens", 0)
                }
            }
        except Exception as e:
            return {
                "decision": options[0],
                "scores": {opt: 1.0 / len(options) for opt in options},
                "latency_ms": 0.0,
                "error": str(e),
                "metadata": {"mode": self.mode}
            }


class SystemQDecisionHeadAdapter(DecisionModel):
    """Adapter for Variant B: System Q Decision Head."""

    def __init__(self, decision_model):
        super().__init__(name="System_Q_DecisionHead", is_remote=False)
        self.model = decision_model

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data["input"]
        task_type = input_data.get("task_type", "binary")
        options = input_data.get("options", ["NO", "YES"])
        try:
            res = self.model.predict(text, task_type=task_type, options=options)
            return {
                "decision": res["decision"],
                "scores": res["scores"],
                "latency_ms": res["latency_ms"],
                "error": None,
                "metadata": {"task_type": task_type}
            }
        except Exception as e:
            return {
                "decision": options[0],
                "scores": {opt: 1.0 / len(options) for opt in options},
                "latency_ms": 0.0,
                "error": str(e),
                "metadata": {"task_type": task_type}
            }


class SystemQEarlyExitAdapter(DecisionModel):
    """Adapter for Variant D: Adaptive Early-Exit Decision Model."""

    def __init__(self, early_exit_model):
        super().__init__(name="System_Q_EarlyExit", is_remote=False)
        self.model = early_exit_model

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data["input"]
        task_type = input_data.get("task_type", "binary")
        options = input_data.get("options", ["NO", "YES"])
        try:
            res = self.model.predict(text, task_type=task_type, options=options)
            return {
                "decision": res["decision"],
                "scores": res["scores"],
                "latency_ms": res["latency_ms"],
                "error": None,
                "metadata": {
                    "exit_layer": res["exit_layer"],
                    "layers_executed": res["layers_executed"],
                    "early_exited": res["early_exited"],
                    "exit_confidence": res["exit_confidence"]
                }
            }
        except Exception as e:
            return {
                "decision": options[0],
                "scores": {opt: 1.0 / len(options) for opt in options},
                "latency_ms": 0.0,
                "error": str(e),
                "metadata": {}
            }


class LayaAdapter(DecisionModel):
    """Adapter for Convai Innovations Laya (ModernBERT-based RL Agent)."""

    def __init__(self, laya_agent):
        super().__init__(name="Laya_Local", is_remote=False)
        self.agent = laya_agent

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data["input"]
        options = input_data["options"]

        questions = {
            "decision": {
                "type": "choice",
                "instructions": "Make the structured decision matching one of the options.",
                "criteria": options
            }
        }

        t0 = time.perf_counter()
        try:
            res = self.agent.predict(text, questions)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            ans = res["answers"]["decision"]
            choice = ans.get("choice", options[0])
            probs = ans.get("probabilities", {opt: 1.0 / len(options) for opt in options})

            # Align probs to options
            scores = {opt: round(float(probs.get(opt, 0.0)), 4) for opt in options}

            return {
                "decision": choice,
                "scores": scores,
                "latency_ms": round(latency_ms, 2),
                "error": None,
                "metadata": {
                    "confidence": ans.get("confidence", 1.0),
                    "model": res.get("model", "laya")
                }
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "decision": options[0],
                "scores": {opt: 1.0 / len(options) for opt in options},
                "latency_ms": round(latency_ms, 2),
                "error": str(e),
                "metadata": {}
            }


class JevAdapter(DecisionModel):
    """
    Adapter for TypeSafe Jev API.
    Honoring research requirements: connects to verified API when TYPESAFE_API_KEY is available;
    records exact status and limitation when key is unavailable without fabricating results.
    """

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(name="TypeSafe_Jev_Remote", is_remote=True)
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.endpoint = "https://api.typesafe.ai/v1/decision"

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        options = input_data["options"]
        if not self.api_key:
            return {
                "decision": "UNAVAILABLE",
                "scores": {opt: 0.0 for opt in options},
                "latency_ms": None,
                "error": "TYPESAFE_API_KEY not configured. Remote service inaccessible.",
                "metadata": {
                    "is_remote": True,
                    "status": "UNCONFIGURED_CREDENTIALS"
                }
            }

        # If key is provided, execute authenticated HTTP request
        import urllib.request
        payload = {
            "model": "jev-latest",
            "state": input_data["input"],
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Select the correct category.",
                    "options": options
                }
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            ans = raw["answers"]["decision"]
            return {
                "decision": ans["choice"],
                "scores": ans["probabilities"],
                "latency_ms": round(latency_ms, 2),
                "error": None,
                "metadata": {
                    "is_remote": True,
                    "service_latency_ms": raw.get("usage", {}).get("execution_ms", None),
                    "network_latency_ms": round(latency_ms - raw.get("usage", {}).get("execution_ms", 0), 2)
                }
            }
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "decision": options[0],
                "scores": {opt: 1.0 / len(options) for opt in options},
                "latency_ms": round(latency_ms, 2),
                "error": f"Jev API request failed: {e}",
                "metadata": {"is_remote": True}
            }


class BenchmarkSuite:
    """Executes comparative benchmark across models on evaluation datasets."""

    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        output_dir: str = "results",
        seed: int = 42
    ):
        self.dataset = dataset
        self.output_dir = output_dir
        self.seed = seed
        os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)

    def run_model_benchmark(
        self,
        model_adapter: DecisionModel,
        warmup_runs: int = 2,
        repeats: int = 1
    ) -> Dict[str, Any]:
        """
        Runs complete benchmark protocol for a model adapter.
        """
        print(f"\n=======================================================")
        print(f"BENCHMARKING: {model_adapter.name} (Remote: {model_adapter.is_remote})")
        print(f"=======================================================")

        # Warmup
        if warmup_runs > 0 and len(self.dataset) > 0:
            print(f"Running {warmup_runs} warmup passes...")
            for i in range(warmup_runs):
                ex = self.dataset[i % len(self.dataset)]
                _ = model_adapter.predict(ex)

        random.seed(self.seed)
        shuffled = list(self.dataset)
        random.shuffle(shuffled)

        predictions = []
        ground_truths = []
        latencies = []
        scores_list = []
        individual_records = []
        num_errors = 0

        # Reset peak GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)

        # Track energy across full evaluation run
        with EnergyContext(label=model_adapter.name, num_decisions=len(shuffled) * repeats) as energy_ctx:
            for rep in range(repeats):
                for item in shuffled:
                    res = model_adapter.predict(item)
                    if res.get("error"):
                        num_errors += 1

                    pred_decision = res["decision"]
                    gt = item["ground_truth"]
                    options = item["options"]

                    predictions.append(pred_decision)
                    ground_truths.append(gt)
                    if res["latency_ms"] is not None:
                        latencies.append(res["latency_ms"])

                    # Build probability vector
                    scores = res.get("scores", {})
                    prob_vec = [scores.get(opt, 0.0) for opt in options]
                    # Normalize if sum > 0
                    s = sum(prob_vec)
                    if s > 0:
                        prob_vec = [p / s for p in prob_vec]
                    else:
                        prob_vec = [1.0 / len(options) for _ in options]
                    scores_list.append(prob_vec)

                    record = {
                        "example_id": item["id"],
                        "task_type": item["task_type"],
                        "ground_truth": gt,
                        "prediction": pred_decision,
                        "is_correct": (pred_decision == gt),
                        "latency_ms": res["latency_ms"],
                        "scores": scores,
                        "error": res.get("error"),
                        "metadata": res.get("metadata", {})
                    }
                    individual_records.append(record)

        # Memory telemetry
        vram_peak_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        ram_peak_gb = psutil.Process().memory_info().rss / (1024 ** 3)

        # Compute accuracy metrics
        correct_count = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
        total_count = len(ground_truths)
        accuracy = correct_count / max(1, total_count)

        # Speed metrics (warm batch-1)
        if latencies:
            lat_arr = np.array(latencies)
            cold_latency = latencies[0]
            mean_latency = float(np.mean(lat_arr))
            median_latency = float(np.median(lat_arr))
            p90_latency = float(np.percentile(lat_arr, 90))
            p95_latency = float(np.percentile(lat_arr, 95))
            throughput = 1000.0 / mean_latency if mean_latency > 0 else 0.0
        else:
            cold_latency = mean_latency = median_latency = p90_latency = p95_latency = throughput = None

        # Energy telemetry
        energy_metrics = energy_ctx.results

        # Per-task breakdown
        per_task = {}
        for rec in individual_records:
            t = rec["task_type"]
            per_task.setdefault(t, {"correct": 0, "total": 0})
            per_task[t]["total"] += 1
            if rec["is_correct"]:
                per_task[t]["correct"] += 1

        for t in per_task:
            per_task[t]["accuracy"] = round(per_task[t]["correct"] / per_task[t]["total"], 4)

        result_summary = {
            "model_name": model_adapter.name,
            "is_remote": model_adapter.is_remote,
            "num_evaluated": total_count,
            "num_errors": num_errors,
            "accuracy": round(accuracy, 4),
            "per_task_accuracy": per_task,
            "speed": {
                "cold_start_ms": round(cold_latency, 2) if cold_latency else None,
                "warm_mean_ms": round(mean_latency, 2) if mean_latency else None,
                "warm_median_ms": round(median_latency, 2) if median_latency else None,
                "p90_ms": round(p90_latency, 2) if p90_latency else None,
                "p95_ms": round(p95_latency, 2) if p95_latency else None,
                "throughput_decisions_per_sec": round(throughput, 2) if throughput else None
            },
            "memory": {
                "peak_vram_gb": round(vram_peak_gb, 3),
                "peak_ram_gb": round(ram_peak_gb, 3)
            },
            "energy": energy_metrics,
            "records": individual_records
        }

        # Save raw results
        out_path = os.path.join(self.output_dir, "raw", f"{model_adapter.name}_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_summary, f, indent=2)

        print(f"Accuracy:        {result_summary['accuracy']:.4f}")
        if mean_latency:
            print(f"Warm Median Lat: {result_summary['speed']['warm_median_ms']} ms (P95: {result_summary['speed']['p95_ms']} ms)")
        print(f"Peak VRAM:       {result_summary['memory']['peak_vram_gb']} GB")
        if energy_metrics.get("measured"):
            print(f"Avg Power / Eng: {energy_metrics['power_avg_w']} W / {energy_metrics['energy_mj_per_decision']} mJ/decision")
        print(f"Results saved:   {out_path}")

        return result_summary
