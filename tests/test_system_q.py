"""
Unit Test Suite for System Q (Direct Candidate Scoring Architecture).
Validates:
- Hardware detection and power telemetry
- Calibration metrics (ECE, Brier, NLL) and temperature scaling
- SystemQDirectScorer canonical anchor formatting, prompt generation, and prediction
- Direct candidate margin loss computation
- Jev adapter unconfigured handling
"""

import os
import pytest
import torch
import numpy as np

from src.hardware_detection import HardwareDetector
from src.energy_monitor import GPUEnergyMonitor, EnergyContext
from src.calibration import (
    compute_ece,
    compute_brier_score,
    compute_nll,
    TemperatureScaler
)
from src.system_q_scorer import SystemQDirectScorer, ANCHOR_SYMBOLS
from src.train_direct_scorer import compute_direct_candidate_loss
from src.benchmark_runner import JevAdapter


class MockCausalLM(torch.nn.Module):
    """Lightweight mock causal language model for fast unit testing."""
    def __init__(self, vocab_size: int = 1000, hidden_size: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        logits = self.lm_head(x)
        class MockOutput:
            pass
        out = MockOutput()
        out.logits = logits
        return out


class MockTokenizer:
    """Mock tokenizer matching Qwen character behavior for tests."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        # Deterministic ascii-based token IDs
        cleaned = text.strip()
        if not cleaned:
            return [10]
        return [ord(cleaned[0]) % 500 + 20]

    def __call__(self, text, return_tensors="pt", max_length=64, truncation=True, padding=True):
        tokens = [ord(c) % 500 + 20 for c in text[:max_length]]
        pad_len = max_length - len(tokens)
        input_ids = tokens + [0] * pad_len
        attention_mask = [1] * len(tokens) + [0] * pad_len
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long)
        }


# -------------------------------------------------------------
# 1. Hardware & Energy Telemetry Tests
# -------------------------------------------------------------
def test_hardware_detector():
    detector = HardwareDetector()
    profile = detector.get_full_profile()
    assert "os" in profile
    assert "cpu" in profile
    assert "gpu" in profile
    assert profile["cpu"]["total_ram_gb"] > 0
    assert profile["safe_batch_size"] >= 1


def test_energy_context():
    with EnergyContext(label="unit_test_energy", sample_interval_ms=10) as ctx:
        import time
        time.sleep(0.04)
    res = ctx.results
    assert "measured" in res
    assert res["wall_time_ms"] > 0
    assert "energy_joules" in res


# -------------------------------------------------------------
# 2. Calibration Metrics Tests
# -------------------------------------------------------------
def test_calibration_ece_and_brier():
    probs = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.2, 0.8]
    ])
    labels = np.array([0, 0, 1, 1])
    ece_res = compute_ece(probs, labels, num_bins=5)
    assert 0.0 <= ece_res["ece"] <= 1.0

    brier = compute_brier_score(probs, labels, num_classes=2)
    assert 0.0 <= brier <= 2.0

    nll = compute_nll(probs, labels)
    assert nll >= 0.0


def test_temperature_scaler():
    logits = torch.tensor([[2.0, -1.0], [1.5, -0.5], [-2.0, 3.0]], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    scaler = TemperatureScaler()
    opt_t = scaler.fit(logits, labels)
    assert opt_t > 0.0
    calibrated = scaler.calibrate(logits)
    assert torch.allclose(calibrated.sum(dim=-1), torch.ones(3), atol=1e-5)


# -------------------------------------------------------------
# 3. System Q Direct Scorer Architecture Tests
# -------------------------------------------------------------
def test_system_q_scorer_formatting_and_anchors():
    model = MockCausalLM()
    tok = MockTokenizer()
    scorer = SystemQDirectScorer(model=model, tokenizer=tok, enable_null_calibration=False)

    options = ["SCAM", "LEGITIMATE"]
    prompt, anchors = scorer._format_prompt("Test phishing text", options)
    assert "(A) SCAM" in prompt
    assert "(B) LEGITIMATE" in prompt
    assert len(anchors) == 2
    assert "Selected Option: (" in prompt


def test_system_q_scorer_prediction_schema():
    model = MockCausalLM()
    tok = MockTokenizer()
    scorer = SystemQDirectScorer(model=model, tokenizer=tok, enable_null_calibration=False)

    options = ["BILLING", "TECHNICAL_SUPPORT", "ACCOUNT_ACCESS"]
    res = scorer.predict("I need a refund for my invoice", options)

    assert "decision" in res
    assert res["decision"] in options
    assert "scores" in res
    assert len(res["scores"]) == 3
    assert abs(sum(res["scores"].values()) - 1.0) < 1e-3
    assert "confidence" in res
    assert "margin" in res
    assert res["generated_tokens"] == 0
    assert res["latency_ms"] >= 0


def test_direct_candidate_loss():
    model = MockCausalLM()
    tok = MockTokenizer()
    device = torch.device("cpu")
    loss = compute_direct_candidate_loss(
        model=model,
        tokenizer=tok,
        input_text="Double charged on card",
        options=["BILLING", "TECHNICAL_SUPPORT"],
        ground_truth="BILLING",
        device=device
    )
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


# -------------------------------------------------------------
# 4. Adapter and Error Handling Tests
# -------------------------------------------------------------
def test_jev_adapter_unconfigured_behavior():
    adapter = JevAdapter(api_key=None)
    res = adapter.predict({
        "input": "Test query",
        "options": ["A", "B"]
    })
    assert res["decision"] == "UNAVAILABLE"
    assert "TYPESAFE_API_KEY" in res["error"]
    assert res["metadata"]["is_remote"] is True
