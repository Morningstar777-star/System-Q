"""
System Q Decision Model (Variant B).
Adapts Qwen2.5-3B backbone with task-specific classification heads attached
to the pooled final hidden representations.
Supports frozen backbone, LoRA adaptation, and probability calibration.
"""

import os
import time
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, TaskType

from src.decision_heads import (
    pool_hidden_states,
    BinaryDecisionHead,
    MultiClassDecisionHead,
    CandidateRankingHead,
    BaseDecisionHead
)


class SystemQDecisionModel(nn.Module):
    """
    Core System 1 Decision Architecture for Qwen2.5-3B.
    Replaces autoregressive causal generation with structured non-autoregressive decision heads.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        hidden_size: int = 2048,
        pooling: str = "last_token"
    ):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.pooling = pooling

        # Register task decision heads
        self.heads = nn.ModuleDict({
            "binary": BinaryDecisionHead(hidden_size, label_names=["NON_URGENT", "URGENT"]),
            "multiclass": MultiClassDecisionHead(hidden_size, label_names=["BILLING", "TECH_SUPPORT", "ACCOUNT", "SALES"]),
            "ranking": CandidateRankingHead(hidden_size),
            "ambiguous": BinaryDecisionHead(hidden_size, label_names=["UNCERTAIN", "RESOLVABLE"]),
            "adversarial": BinaryDecisionHead(hidden_size, label_names=["NON_URGENT", "URGENT"]),
        })
        try:
            device = next(self.backbone.parameters()).device
            self.heads.to(device)
        except Exception:
            pass

    def freeze_backbone(self):
        """Freezes all backbone transformer weights for stage 1 head-only tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone transformer parameters frozen.")

    def enable_lora(self, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
        """Applies LoRA adapters to attention projection layers for stage 2 tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=lora_dropout,
            bias="none"
        )
        self.backbone = get_peft_model(self.backbone, lora_config)
        print(f"LoRA enabled on backbone (r={r}, alpha={lora_alpha}). Trainable parameters:")
        self.backbone.print_trainable_parameters()

    def get_last_hidden_state(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Runs backbone forward pass and extracts the final layer representations."""
        # Check if backbone is CausalLM or base model
        base_model = getattr(self.backbone, "model", self.backbone)
        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_type: str = "binary"
    ) -> Dict[str, Any]:
        """
        Executes single forward pass through backbone and applies selected task head.
        """
        last_hidden_state = self.get_last_hidden_state(input_ids, attention_mask)
        pooled = pool_hidden_states(last_hidden_state, attention_mask, pooling=self.pooling)

        if task_type not in self.heads:
            # Fallback to binary or multiclass
            task_type = "multiclass" if "multiclass" in self.heads else "binary"

        head = self.heads[task_type]
        head_param = next(head.parameters())
        pooled = pooled.to(device=head_param.device, dtype=head_param.dtype)
        head_outputs = head(pooled)
        head_outputs["pooled_features"] = pooled
        return head_outputs

    def predict(
        self,
        input_text: str,
        task_type: str = "binary",
        options: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Unified prediction API returning normalized System Q decision schema.
        """
        t0 = time.perf_counter()
        device = next(self.backbone.parameters()).device

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_tokens = input_ids.shape[1]

        with torch.inference_mode():
            outputs = self.forward(input_ids, attention_mask, task_type=task_type)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        decision = outputs["decisions"][0]
        scores = outputs["scores"][0]

        # Filter or align scores to requested options if specified
        if options and all(opt in scores for opt in options):
            scores = {opt: scores[opt] for opt in options}

        return {
            "decision": decision,
            "scores": scores,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "task_type": task_type,
            "error": None
        }

    def save_heads(self, save_dir: str):
        """Saves weights of all decision heads."""
        os.makedirs(save_dir, exist_ok=True)
        head_path = os.path.join(save_dir, "decision_heads.pt")
        torch.save(self.heads.state_dict(), head_path)
        print(f"Decision heads saved to {head_path}")

    def load_heads(self, load_dir: str):
        """Loads weights of all decision heads."""
        head_path = os.path.join(load_dir, "decision_heads.pt")
        if os.path.exists(head_path):
            device = next(self.parameters()).device
            self.heads.load_state_dict(torch.load(head_path, map_location=device))
            print(f"Decision heads loaded from {head_path}")
        else:
            print(f"Warning: Heads file {head_path} not found.")
