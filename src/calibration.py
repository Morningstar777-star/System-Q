"""
Calibration and Uncertainty Quantification for System Q.
Implements Expected Calibration Error (ECE), Brier Score, NLL,
Temperature Scaling optimizer, and Selective Risk vs Coverage analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    using uniform confidence binning.
    
    Args:
        probs: Array of predicted probability distributions, shape (N, num_classes)
        labels: Array of true class indices, shape (N,)
        num_bins: Number of equal-width confidence bins in [0, 1]
    
    Returns:
        Dictionary with ece, mce, and per-bin statistics.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_stats = []
    n = len(labels)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            diff = np.abs(accuracy_in_bin - avg_confidence_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            bin_stats.append({
                "bin": f"[{bin_lower:.2f}, {bin_upper:.2f}]",
                "count": int(np.sum(in_bin)),
                "acc": round(float(accuracy_in_bin), 4),
                "conf": round(float(avg_confidence_in_bin), 4),
                "gap": round(float(diff), 4)
            })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "num_bins": num_bins,
        "bin_details": bin_stats
    }


def compute_brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
    num_classes: Optional[int] = None
) -> float:
    """Computes standard multi-class Brier quadratic scoring rule."""
    n = len(labels)
    if num_classes is None:
        num_classes = probs.shape[1]
    one_hot = np.zeros((n, num_classes))
    for i, l in enumerate(labels):
        one_hot[i, int(l)] = 1.0
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return round(float(brier), 4)


def compute_nll(
    probs: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-12
) -> float:
    """Computes Negative Log-Likelihood (cross entropy) over predicted probabilities."""
    n = len(labels)
    clipped_probs = np.clip(probs, eps, 1.0)
    log_likelihood = 0.0
    for i, l in enumerate(labels):
        log_likelihood += np.log(clipped_probs[i, int(l)])
    nll = - (log_likelihood / n)
    return round(float(nll), 4)


class TemperatureScaler:
    """Learns post-hoc temperature parameter T on validation logits to minimize NLL."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 100, lr: float = 0.01) -> float:
        """
        Tunes temperature T > 0 on validation set.
        """
        t_param = nn.Parameter(torch.ones(1, device=val_logits.device))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([t_param], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled = val_logits / torch.clamp(t_param, min=0.01)
            loss = criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature = max(0.01, float(t_param.item()))
        return round(self.temperature, 4)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits and applies softmax."""
        scaled = logits / self.temperature
        return torch.softmax(scaled, dim=-1)


def evaluate_selective_risk(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
) -> List[Dict[str, Any]]:
    """
    Evaluates selective classification: accuracy vs coverage vs abstention rate
    across varying confidence thresholds.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)
    n = len(labels)

    curve = []
    for tau in thresholds:
        accepted = confidences >= tau
        coverage = np.mean(accepted)
        abstention_rate = 1.0 - coverage

        if np.sum(accepted) > 0:
            acc_at_tau = np.mean(correct[accepted])
            risk_at_tau = 1.0 - acc_at_tau
        else:
            acc_at_tau = 1.0
            risk_at_tau = 0.0

        curve.append({
            "threshold": tau,
            "coverage": round(float(coverage), 4),
            "abstention_rate": round(float(abstention_rate), 4),
            "selective_accuracy": round(float(acc_at_tau), 4),
            "selective_risk": round(float(risk_at_tau), 4),
            "accepted_count": int(np.sum(accepted)),
            "total_count": n
        })

    return curve
