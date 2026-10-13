"""
Task-Specific Fine-Tuning Pipeline for System Q Direct Candidate Scoring.
Implements:
1. Direct Candidate Cross-Entropy over Canonical Symbol Anchors
2. Contrastive Margin Ranking Loss: Penalizes rival candidates within margin safety bounds
3. PEFT LoRA on Attention & MLP projections with optional post-training weight fusion
"""

import os
import json
import time
import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from src.system_q_scorer import SystemQDirectScorer, ANCHOR_SYMBOLS, DOMAIN_DESCRIPTIONS


def compute_direct_candidate_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_text: str,
    options: List[str],
    ground_truth: str,
    device: torch.device,
    temperature: float = 0.8,
    margin_target: float = 2.5,
    margin_weight: float = 0.5
) -> torch.Tensor:
    """
    Computes hybrid loss:
    L = L_CE(cand_logits / tau, target) + lambda * relu(margin_target - (z_target - max_{j != target} z_j))
    """
    lines = ["You are a fast, precise decision engine. Analyze the input and choose exactly one option."]
    lines.append("Allowed Options:")
    anchor_tokens = []
    target_idx = 0
    for i, opt in enumerate(options):
        sym = ANCHOR_SYMBOLS[i]
        desc = DOMAIN_DESCRIPTIONS.get(opt, opt)
        lines.append(f"({sym}) {opt}: {desc}")
        tok_id = tokenizer.encode(sym, add_special_tokens=False)[0]
        anchor_tokens.append(tok_id)
        if opt == ground_truth:
            target_idx = i

    menu = "\n".join(lines)
    prompt = (
        f"<|im_start|>system\n{menu}<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        f"<|im_start|>assistant\nSelected Option: ("
    )

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    last_logits = outputs.logits[0, -1, :]

    cand_logits = [last_logits[tok_id] for tok_id in anchor_tokens]
    cand_tensor = torch.stack(cand_logits).unsqueeze(0)

    # 1. Direct candidate cross-entropy
    ce_loss = F.cross_entropy(cand_tensor / temperature, torch.tensor([target_idx], device=device))

    # 2. Contrastive margin ranking penalty
    if len(options) > 1:
        target_logit = cand_tensor[0, target_idx]
        other_indices = [i for i in range(len(options)) if i != target_idx]
        max_distractor = torch.max(cand_tensor[0, other_indices])
        margin_penalty = F.relu(margin_target - (target_logit - max_distractor))
        total_loss = ce_loss + margin_weight * margin_penalty
    else:
        total_loss = ce_loss

    return total_loss


def train_direct_scorer_lora(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    training_data: List[Dict[str, Any]],
    output_adapter_dir: str = "models/adapters/system_q_sota_direct_scorer",
    epochs: int = 4,
    lr: float = 2e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    target_modules: Optional[List[str]] = None,
    gradient_accumulation_steps: int = 4,
    merge_weights_after_training: bool = False
) -> PreTrainedModel:
    """
    Applies LoRA to Qwen2.5-3B and trains with Contrastive Candidate Margin Loss.
    Optionally merges weights into base model for zero-overhead inference.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    print(f"\n--- Initializing SOTA LoRA on Qwen2.5-3B (r={lora_r}, alpha={lora_alpha}) ---")
    print(f"Target Modules: {target_modules}")
    os.makedirs(output_adapter_dir, exist_ok=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )

    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    device = next(lora_model.parameters()).device
    total_steps = (len(training_data) // gradient_accumulation_steps) * epochs

    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )

    lora_model.train()
    print(f"Beginning SOTA Direct Candidate Training ({len(training_data)} examples, {epochs} epochs)...")

    global_step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        total_loss = 0.0
        optimizer.zero_grad()

        # Shuffle training examples per epoch
        permuted_data = [training_data[i] for i in torch.randperm(len(training_data)).tolist()]

        for step, item in enumerate(permuted_data):
            loss = compute_direct_candidate_loss(
                lora_model,
                tokenizer,
                input_text=item["input"],
                options=item["options"],
                ground_truth=item["ground_truth"],
                device=device,
                temperature=0.8,
                margin_target=2.5,
                margin_weight=0.5
            )

            loss = loss / gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item() * gradient_accumulation_steps

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(permuted_data):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = total_loss / len(training_data)
        epoch_time = time.perf_counter() - t0
        print(f"Epoch {epoch}/{epochs} | Avg Margin Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

    # Save adapter
    lora_model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    print(f"SOTA LoRA Adapter successfully saved to: {output_adapter_dir}")
    lora_model.eval()

    if merge_weights_after_training:
        print("Fusing LoRA weights directly into base INT4 model for SOTA inference speed...")
        merged_model = lora_model.merge_and_unload()
        merged_model.eval()
        return merged_model

    return lora_model


if __name__ == "__main__":
    print("SOTA Direct scorer training script ready.")
