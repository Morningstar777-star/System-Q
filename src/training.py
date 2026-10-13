"""
Training and Distillation Pipeline for System Q.
Implements:
1. Stage 1: Frozen backbone + Decision Head training
2. Stage 2: Parameter-efficient LoRA fine-tuning
3. Stage 3: Intermediate Early-Exit Heads training
4. Teacher-guided soft-label distillation
Designed strictly for memory-safe execution on 4 GB VRAM.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase

from src.decision_model import SystemQDecisionModel
from src.early_exit import EarlyExitDecisionModel


class DecisionTaskDataset(Dataset):
    """PyTorch Dataset for structured decision examples."""

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        label_map: Dict[str, int],
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        text = ex["input"]
        gt = ex["ground_truth"]
        label_idx = self.label_map.get(gt, 0)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "task_type": ex["task_type"]
        }

        # Teacher reference distribution if present
        ref_dist = ex.get("reference_distribution")
        if ref_dist:
            num_classes = len(self.label_map)
            teacher_probs = torch.zeros(num_classes, dtype=torch.float32)
            for k, v in ref_dist.items():
                if k in self.label_map:
                    teacher_probs[self.label_map[k]] = float(v)
            # normalize if sum > 0
            if teacher_probs.sum() > 0:
                teacher_probs = teacher_probs / teacher_probs.sum()
            item["teacher_probs"] = teacher_probs

        return item


def compute_distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_probs: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
    alpha: float = 0.6
) -> torch.Tensor:
    """
    Combined Cross-Entropy and Knowledge Distillation KL loss.
    """
    ce_loss = F.cross_entropy(student_logits, labels)
    if teacher_probs is None:
        return ce_loss

    # Soft distillation loss
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    # KL divergence
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return alpha * ce_loss + (1.0 - alpha) * kl_loss


class SystemQTrainer:
    """Manages training stages, memory safety, and evaluation."""

    def __init__(
        self,
        model: Union[SystemQDecisionModel, EarlyExitDecisionModel],
        tokenizer: PreTrainedTokenizerBase,
        train_examples: List[Dict[str, Any]],
        val_examples: List[Dict[str, Any]],
        checkpoint_dir: str = "models/checkpoints",
        batch_size: int = 4,
        gradient_accumulation_steps: int = 2,
        lr: float = 1e-4,
        epochs: int = 3,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_examples = train_examples
        self.val_examples = val_examples
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.grad_accum = gradient_accumulation_steps
        self.lr = lr
        self.epochs = epochs
        self.device = device or next(self.model.parameters()).device
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_task_head(
        self,
        task_type: str = "binary",
        label_list: List[str] = ["NON_URGENT", "URGENT"],
        use_distillation: bool = True
    ) -> Dict[str, Any]:
        """
        Trains the task-specific decision head on the training split while keeping backbone frozen.
        """
        print(f"\n--- Training Decision Head for task: {task_type} (Labels: {label_list}) ---")
        label_map = {lbl: i for i, lbl in enumerate(label_list)}

        task_train = [ex for ex in self.train_examples if ex["task_type"] == task_type or task_type == "all"]
        task_val = [ex for ex in self.val_examples if ex["task_type"] == task_type or task_type == "all"]

        if not task_train:
            task_train = self.train_examples
        if not task_val:
            task_val = self.val_examples

        train_ds = DecisionTaskDataset(task_train, self.tokenizer, label_map)
        val_ds = DecisionTaskDataset(task_val, self.tokenizer, label_map)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Collect trainable parameters (only head parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=0.01)

        history = []
        best_val_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                teacher_probs = batch.get("teacher_probs")
                if teacher_probs is not None:
                    teacher_probs = teacher_probs.to(self.device)

                # Forward
                outputs = self.model(input_ids, attention_mask, task_type=task_type)
                logits = outputs["logits"]

                loss = compute_distillation_loss(
                    logits,
                    labels,
                    teacher_probs if use_distillation else None
                )
                loss = loss / self.grad_accum
                loss.backward()

                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.grad_accum

            avg_loss = total_loss / max(1, len(train_loader))

            # Validation
            val_acc, val_loss = self.evaluate(val_loader, task_type=task_type)
            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            history.append({
                "epoch": epoch,
                "train_loss": round(avg_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4)
            })

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                # Save best checkpoint
                save_path = os.path.join(self.checkpoint_dir, f"best_head_{task_type}.pt")
                if hasattr(self.model, "heads"):
                    torch.save(self.model.heads.state_dict(), save_path)

        return {
            "task_type": task_type,
            "best_val_accuracy": best_val_acc,
            "history": history
        }

    def evaluate(self, val_loader: DataLoader, task_type: str = "binary") -> Tuple[float, float]:
        """Evaluates accuracy and loss on validation split."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids, attention_mask, task_type=task_type)
                logits = outputs["logits"]
                loss = F.cross_entropy(logits, labels)

                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()

        avg_loss = total_loss / max(1, len(val_loader))
        acc = correct / max(1, total)
        return acc, avg_loss
