"""
Baseline Inference Implementations for Variant A (Original Qwen2.5-3B).
Implements:
1. Constrained Short-Answer Generation
2. Direct Candidate Scoring using next-token log-probabilities
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class QwenConstrainedGenerationBaseline:
    """Variant A1: Causal LM generating constrained short-answer decisions."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, max_new_tokens: int = 4):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def predict(self, input_text: str, options: List[str]) -> Dict[str, Any]:
        """
        Executes constrained generation and parses output into decision.
        """
        prompt = (
            f"You are a fast decision engine. Analyze the following scenario and choose exactly one of: {options}.\n"
            f"Scenario: {input_text}\n"
            f"Decision:"
        )

        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        generated_ids = outputs[0][prompt_tokens:]
        gen_tokens_count = len(generated_ids)
        gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip().upper()

        # Parse decision matching options
        chosen = options[0]
        for opt in options:
            if opt.upper() in gen_text or gen_text in opt.upper():
                chosen = opt
                break

        # Uniform or pseudo-certain distribution for generation
        scores = {opt: (1.0 if opt == chosen else 0.0) for opt in options}

        return {
            "decision": chosen,
            "raw_generation": gen_text,
            "scores": scores,
            "latency_ms": round(latency_ms, 2),
            "generated_tokens": gen_tokens_count,
            "prompt_tokens": prompt_tokens,
            "error": None
        }


class QwenDirectCandidateScoringBaseline:
    """Variant A2: Direct scoring of candidate labels using next-token logits without text generation."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, input_text: str, options: List[str]) -> Dict[str, Any]:
        """
        Executes a single forward pass and scores candidate label tokens directly.
        """
        prompt = (
            f"Context: {input_text}\n"
            f"Allowed choices: {', '.join(options)}.\n"
            f"Selected choice:"
        )

        t0 = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self.model(**inputs)
            # Logits of next token at final sequence position
            next_token_logits = outputs.logits[0, -1, :]

        # Extract logits for each candidate option (using first subtoken with leading space and without)
        candidate_logits = []
        for opt in options:
            # Tokenize option with leading space and without
            tok_ids_sp = self.tokenizer.encode(" " + opt, add_special_tokens=False)
            tok_ids_raw = self.tokenizer.encode(opt, add_special_tokens=False)
            
            val_sp = next_token_logits[tok_ids_sp[0]].item() if tok_ids_sp else -1e4
            val_raw = next_token_logits[tok_ids_raw[0]].item() if tok_ids_raw else -1e4
            candidate_logits.append(max(val_sp, val_raw))

        cand_tensor = torch.tensor(candidate_logits, dtype=torch.float32)
        probs = F.softmax(cand_tensor, dim=-1).tolist()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        scores = {opt: round(probs[i], 4) for i, opt in enumerate(options)}
        best_idx = int(torch.argmax(cand_tensor).item())
        best_decision = options[best_idx]

        return {
            "decision": best_decision,
            "scores": scores,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": 0,
            "error": None
        }
