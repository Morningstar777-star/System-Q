"""
System Q: Ultra-Simple, Developer-Friendly Python API.

Usage:
    from src.system_q import SystemQ

    q = SystemQ()

    # Generic Decision:
    decision = q.decide("Customer wants to cancel subscription", ["RETAIN", "CANCEL", "ESCALATE"])
    print(decision.choice, decision.confidence)

    # Built-in Helpers:
    scam_check = q.is_scam("Click here to claim your $1,000 lottery winnings!")
    print(scam_check.is_scam, scam_check.confidence)

    intent = q.get_intent("My credit card was charged twice for order #4021")
    print(intent.category, intent.confidence)
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from peft import PeftModel

from src.model_loader import load_qwen_model_and_tokenizer
from src.system_q_scorer import SystemQDirectScorer


# Standard Option Presets
SCAM_OPTIONS = ["SCAM", "LEGITIMATE"]
INTENT_OPTIONS = [
    "BILLING",
    "TECHNICAL_SUPPORT",
    "ACCOUNT_ACCESS",
    "SALES_INQUIRY",
    "ACCOUNT_CANCELLATION"
]


@dataclass
class Decision:
    """Structured decision output from System Q."""
    choice: str
    confidence: float
    margin: float
    scores: Dict[str, float]
    latency_ms: float
    anchor: str

    @property
    def is_scam(self) -> bool:
        """True if the decision is 'SCAM'."""
        return self.choice.upper() == "SCAM"

    @property
    def is_legitimate(self) -> bool:
        """True if the decision is 'LEGITIMATE'."""
        return self.choice.upper() == "LEGITIMATE"

    @property
    def category(self) -> str:
        """Alias for choice (convenient for intent classification)."""
        return self.choice

    def __repr__(self) -> str:
        return (
            f"Decision(choice='{self.choice}', "
            f"confidence={self.confidence:.1%}, "
            f"margin={self.margin:.3f}, "
            f"latency={self.latency_ms:.1f}ms)"
        )


class SystemQ:
    """
    High-level, developer-friendly interface to System Q.
    Handles INT4 model loading, LoRA weight fusion, prior calibration, and warmup automatically.
    """

    def __init__(
        self,
        adapter_path: Optional[str] = None,
        precision: str = "4bit",
        calibration_weight: float = 1.25,
        auto_warmup: bool = True
    ):
        """
        Initializes System Q.
        
        Args:
            adapter_path: Path to fine-tuned LoRA adapter. Defaults to SOTA checkpoint.
            precision: Model precision ('4bit' recommended for RTX 3050 / 4GB VRAM).
            calibration_weight: Context-free prior de-biasing weight (default 1.25).
            auto_warmup: Pre-warms CUDA kernels and caches null priors immediately.
        """
        if adapter_path is None:
            # Locate default SOTA adapter relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(base_dir, "models", "adapters", "system_q_sota_direct_scorer")
            if os.path.exists(default_path):
                adapter_path = default_path

        print(f"Initializing System Q ({precision} precision)...", flush=True)
        self.tokenizer, base_model, _ = load_qwen_model_and_tokenizer(precision=precision)

        if adapter_path and os.path.exists(adapter_path):
            print(f"Loading & fusing SOTA adapter from: {adapter_path}...", flush=True)
            peft_model = PeftModel.from_pretrained(base_model, adapter_path)
            self.model = peft_model.merge_and_unload()
            self.model.eval()
        else:
            print("Running base zero-shot backbone without adapter...", flush=True)
            self.model = base_model

        self.scorer = SystemQDirectScorer(
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=1.0,
            null_calibration_weight=calibration_weight,
            enable_null_calibration=True
        )

        if auto_warmup:
            self.scorer.warmup([SCAM_OPTIONS, INTENT_OPTIONS])

        print("System Q is ready for instant decisions.", flush=True)

    def decide(self, text: str, options: List[str]) -> Decision:
        """
        Makes a structured decision across any candidate options in a single forward pass.
        
        Args:
            text: The input text, email body, query, or transaction to analyze.
            options: List of allowed choices, e.g. ['YES', 'NO'] or ['APPROVE', 'REJECT'].
            
        Returns:
            Decision dataclass with choice, confidence, margin, scores, and latency.
        """
        if not options:
            raise ValueError("options list cannot be empty.")
        if len(options) == 1:
            return Decision(
                choice=options[0],
                confidence=1.0,
                margin=1.0,
                scores={options[0]: 1.0},
                latency_ms=0.0,
                anchor="A"
            )

        res = self.scorer.predict(text, options)
        return Decision(
            choice=res["decision"],
            confidence=res["confidence"],
            margin=res["margin"],
            scores=res["scores"],
            latency_ms=res["latency_ms"],
            anchor=res["anchor"]
        )

    def is_scam(self, email_or_message: str) -> Decision:
        """
        Specialized helper for Scam / Phishing detection.
        
        Returns:
            Decision where .is_scam evaluates to True/False.
        """
        return self.decide(email_or_message, SCAM_OPTIONS)

    def get_intent(self, customer_message: str) -> Decision:
        """
        Specialized helper for Email Intent Classification & Customer Routing.
        Choices: BILLING, TECHNICAL_SUPPORT, ACCOUNT_ACCESS, SALES_INQUIRY, ACCOUNT_CANCELLATION.
        
        Returns:
            Decision where .category contains the routing intent.
        """
        return self.decide(customer_message, INTENT_OPTIONS)
