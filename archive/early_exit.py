"""
Adaptive Early-Exit Decision Model for System Q (Variant D).
Attaches auxiliary decision heads to intermediate transformer layers (e.g. 12, 24, 36).
Exits early when confidence or uncertainty criteria are satisfied, measuring layer savings and speedup.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.decision_heads import (
    pool_hidden_states,
    BinaryDecisionHead,
    MultiClassDecisionHead
)


class EarlyExitDecisionModel(nn.Module):
    """
    Progressively evaluates intermediate transformer blocks and exits as soon as
    decision confidence satisfies the policy threshold.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hidden_size: int = 2048,
        exit_layers: List[int] = [12, 24, 36],
        policy: str = "confidence",
        confidence_threshold: float = 0.85,
        entropy_threshold: float = 0.35,
        margin_threshold: float = 1.5,
        pooling: str = "last_token"
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.exit_layers = sorted(exit_layers)
        self.policy = policy
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold
        self.pooling = pooling
        self.early_exit_enabled = True

        # Attach auxiliary heads at each exit layer for binary and multiclass tasks
        self.exit_heads = nn.ModuleDict()
        for layer in self.exit_layers:
            self.exit_heads[f"layer_{layer}_binary"] = BinaryDecisionHead(
                hidden_size, label_names=["NON_URGENT", "URGENT"]
            )
            self.exit_heads[f"layer_{layer}_multiclass"] = MultiClassDecisionHead(
                hidden_size, label_names=["BILLING", "TECH_SUPPORT", "ACCOUNT", "SALES"]
            )
        try:
            device = next(backbone.parameters()).device
            self.exit_heads.to(device)
        except Exception:
            pass

    def should_exit(self, probs: torch.Tensor, logits: torch.Tensor) -> Tuple[bool, float]:
        """Evaluates whether prediction satisfies the early-exit criterion."""
        if not self.early_exit_enabled:
            return False, 0.0

        max_prob = torch.max(probs, dim=-1).values.item()

        if self.policy == "confidence":
            return (max_prob >= self.confidence_threshold), max_prob

        elif self.policy == "entropy":
            eps = 1e-12
            entropy = - torch.sum(probs * torch.log(probs + eps), dim=-1).item()
            return (entropy <= self.entropy_threshold), max_prob

        elif self.policy == "margin":
            top2 = torch.topk(logits, k=min(2, logits.size(-1)), dim=-1).values
            margin = (top2[0, 0] - top2[0, 1]).item()
            return (margin >= self.margin_threshold), max_prob

        return False, max_prob

    def forward_progressive(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_type: str = "binary"
    ) -> Dict[str, Any]:
        base_model = getattr(self.backbone, "model", self.backbone)
        device = input_ids.device
        
        # Forward through model obtaining hidden states across all layers
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        all_hidden_states = outputs.hidden_states  # tuple of length 37 (embeds + 36 layers)
        total_layers = len(all_hidden_states) - 1

        for exit_layer in self.exit_layers:
            target_idx = min(exit_layer, total_layers)
            hidden_at_exit = all_hidden_states[target_idx]

            # Normalize and pool representation at current exit
            norm_layer = base_model.norm
            normed_hidden = norm_layer(hidden_at_exit)
            pooled = pool_hidden_states(normed_hidden, attention_mask, pooling=self.pooling)

            # Evaluate auxiliary head for this layer
            head_key = f"layer_{exit_layer}_{task_type}"
            if head_key not in self.exit_heads:
                head_key = f"layer_{exit_layer}_multiclass" if "multiclass" in task_type else f"layer_{exit_layer}_binary"

            head = self.exit_heads[head_key]
            head_param = next(head.parameters())
            pooled = pooled.to(device=head_param.device, dtype=head_param.dtype)
            head_out = head(pooled)
            probs = head_out["probabilities"]
            logits = head_out["logits"]

            is_final_exit = (exit_layer == self.exit_layers[-1])
            can_exit, conf = self.should_exit(probs, logits)

            if can_exit or is_final_exit:
                return {
                    "exit_layer": exit_layer,
                    "layers_executed": exit_layer,
                    "confidence": round(conf, 4),
                    "early_exited": (exit_layer < self.exit_layers[-1]),
                    "decisions": head_out["decisions"],
                    "scores": head_out["scores"],
                    "logits": logits,
                    "probabilities": probs,
                    "pooled_features": pooled
                }

        # Fallback to final exit
        return {
            "exit_layer": total_layers,
            "layers_executed": total_layers,
            "confidence": 1.0,
            "early_exited": False,
            "decisions": head_out["decisions"],
            "scores": head_out["scores"],
            "logits": logits,
            "probabilities": probs,
            "pooled_features": pooled
        }

    def predict(
        self,
        input_text: str,
        task_type: str = "binary",
        options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Runs progressive early-exit inference and measures execution telemetry."""
        t0 = time.perf_counter()
        device = next(self.parameters()).device

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_tokens = input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.forward_progressive(input_ids, attention_mask, task_type=task_type)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        decision = outputs["decisions"][0]
        scores = outputs["scores"][0]

        if options and all(opt in scores for opt in options):
            scores = {opt: scores[opt] for opt in options}

        return {
            "decision": decision,
            "scores": scores,
            "latency_ms": round(latency_ms, 2),
            "exit_layer": outputs["exit_layer"],
            "layers_executed": outputs["layers_executed"],
            "early_exited": outputs["early_exited"],
            "exit_confidence": outputs["confidence"],
            "prompt_tokens": prompt_tokens,
            "task_type": task_type,
            "error": None
        }
