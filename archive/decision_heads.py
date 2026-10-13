"""
Decision Heads and Pooling Implementations for System Q.
Transforms hidden representations into structured classification decisions and calibrated probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pooling: str = "last_token"
) -> torch.Tensor:
    """
    Pools sequence representations into a single fixed-dimensional vector per batch item.
    
    Args:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_size)
        attention_mask: Tensor of shape (batch_size, seq_len), 1 for valid tokens, 0 for padding
        pooling: "last_token" (canonical for causal LMs), "mean", or "first_token"
    
    Returns:
        Tensor of shape (batch_size, hidden_size)
    """
    batch_size, seq_len, hidden_size = hidden_states.shape

    if attention_mask is None:
        if pooling == "last_token":
            return hidden_states[:, -1, :]
        elif pooling == "mean":
            return hidden_states.mean(dim=1)
        elif pooling == "first_token":
            return hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

    if pooling == "last_token":
        # In causal LMs, find the index of the last non-padding token
        # sequence_lengths = attention_mask.sum(dim=1) - 1
        seq_lengths = attention_mask.long().sum(dim=1) - 1
        seq_lengths = torch.clamp(seq_lengths, min=0, max=seq_len - 1)
        # Gather the last non-padded token
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        return hidden_states[batch_indices, seq_lengths]

    elif pooling == "mean":
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    elif pooling == "first_token":
        return hidden_states[:, 0, :]

    else:
        raise ValueError(f"Unknown pooling method: {pooling}")


class BaseDecisionHead(nn.Module):
    """Abstract base class for System Q task decision heads."""
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, pooled_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class BinaryDecisionHead(BaseDecisionHead):
    """
    Binary decision head outputting calibrated 2-class probabilities and decisions.
    """
    def __init__(self, hidden_size: int, label_names: Optional[List[str]] = None, dropout: float = 0.1):
        super().__init__(hidden_size, dropout)
        self.labels = label_names or ["NO", "YES"]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            self.dropout,
            nn.Linear(hidden_size // 4, 2)
        )

    def forward(self, pooled_features: torch.Tensor) -> Dict[str, Any]:
        logits = self.classifier(pooled_features)
        scaled_logits = logits / torch.clamp(self.temperature, min=0.01)
        probs = F.softmax(scaled_logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1)

        decisions = [self.labels[idx.item()] for idx in pred_idx]
        score_dicts = []
        for b in range(probs.size(0)):
            score_dicts.append({self.labels[i]: round(probs[b, i].item(), 4) for i in range(len(self.labels))})

        return {
            "logits": logits,
            "scaled_logits": scaled_logits,
            "probabilities": probs,
            "predicted_index": pred_idx,
            "decisions": decisions,
            "scores": score_dicts
        }


class MultiClassDecisionHead(BaseDecisionHead):
    """
    Multi-class decision head supporting arbitrary number of categorical classes.
    """
    def __init__(self, hidden_size: int, label_names: List[str], dropout: float = 0.1):
        super().__init__(hidden_size, dropout)
        self.labels = label_names
        self.num_classes = len(label_names)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            self.dropout,
            nn.Linear(hidden_size // 4, self.num_classes)
        )

    def forward(self, pooled_features: torch.Tensor) -> Dict[str, Any]:
        logits = self.classifier(pooled_features)
        scaled_logits = logits / torch.clamp(self.temperature, min=0.01)
        probs = F.softmax(scaled_logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1)

        decisions = [self.labels[idx.item()] for idx in pred_idx]
        score_dicts = []
        for b in range(probs.size(0)):
            score_dicts.append({self.labels[i]: round(probs[b, i].item(), 4) for i in range(self.num_classes)})

        return {
            "logits": logits,
            "scaled_logits": scaled_logits,
            "probabilities": probs,
            "predicted_index": pred_idx,
            "decisions": decisions,
            "scores": score_dicts
        }


class CandidateRankingHead(BaseDecisionHead):
    """
    Candidate selection and scoring head. Projects representation into scalar choice score.
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__(hidden_size, dropout)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            self.dropout,
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, candidate_pooled_features: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            candidate_pooled_features: Tensor of shape (batch_size, num_candidates, hidden_size)
        """
        scores = self.scorer(candidate_pooled_features).squeeze(-1) # (batch_size, num_candidates)
        probs = F.softmax(scores / torch.clamp(self.temperature, min=0.01), dim=-1)
        best_idx = torch.argmax(probs, dim=-1)
        return {
            "scores": scores,
            "probabilities": probs,
            "best_index": best_idx
        }
