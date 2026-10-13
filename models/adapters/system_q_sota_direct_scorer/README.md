---
base_model: Qwen/Qwen2.5-3B
library_name: peft
pipeline_tag: text-classification
license: apache-2.0
language:
- en
tags:
- decision-making
- text-classification
- scam-detection
- phishing-detection
- intent-classification
- qwen
- lora
- 4bit
- cognitive-architecture
- system-1
widget:
- text: "Urgent: Your account password will expire in 1 hour. Verify credentials immediately at http://suspicious-login.com"
- text: "Could you please send over an updated VAT invoice for our September team subscription?"
---

# ⚡ System Q: Fast System 1 Decision Architecture from Qwen2.5-3B

**System Q** is a non-autoregressive decision model adapted from **Qwen2.5-3B** using **Direct Candidate Scoring**, **Canonical Symbol Anchoring**, and **Hybrid Contrastive Margin Loss**.

Rather than generating paragraphs of text or multi-token strings, System Q evaluates discrete decision spaces in **exactly 1 forward pass** with **0 tokens generated**, achieving **88.0% accuracy** at **~209 ms latency** on consumer 4GB VRAM hardware.

- **GitHub Repository:** [https://github.com/Morningstar777-star/System-Q](https://github.com/Morningstar777-star/System-Q)
- **Live Interactive Demo:** [https://huggingface.co/spaces/LuciferMorningstar777/system-q-demo](https://huggingface.co/spaces/LuciferMorningstar777/system-q-demo)
- **Developer:** [LuciferMorningstar777](https://huggingface.co/LuciferMorningstar777)

---

## 🏆 Head-to-Head Benchmark vs Laya (100 Questions)

Evaluated under identical conditions on an **NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM)**:

| Model Configuration | Architecture | Overall Acc (100 Qs) | Scam Detection (50 Qs) | Email Intent (50 Qs) | Median Latency | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Laya (ModernBERT-large RL)** | Local (Encoder 421M) | **67.0%** | 82.0% | 52.0% | **43.93 ms** | 2.262 GB |
| **⚡ System Q SOTA (Ours)** | Local (INT4 Fused + LoRA) | **88.0%** | **88.0%** | **88.0%** | **209.97 ms** | **2.003 GB** |

- **+21.0% Superiority:** Outperforms Laya by 21 percentage points overall.
- **+36.0% on Intent Categorization:** Solves fine-grained intent boundaries where smaller encoders fail.
- **Sub-250ms Interactive Latency:** Zero autoregressive text generation overhead.
- **Low Memory Footprint:** 2.003 GB peak VRAM (fits on any laptop GPU).

---

## 🚀 Quickstart in 2 Lines of Code

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# 1. Load base Qwen2.5-3B in INT4
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

# 2. Attach System Q SOTA Adapter and fuse weights
peft_model = PeftModel.from_pretrained(base_model, "LuciferMorningstar777/system-q-3b-decision")
model = peft_model.merge_and_unload()
model.eval()

# (Alternatively, clone github.com/Morningstar777-star/System-Q for the 2-line high-level Python API)
```

Using the System Q Python API:
```python
from src.system_q import SystemQ

q = SystemQ()

# Scam Detection
res = q.is_scam("Urgent: Your account has been suspended! Verify at bit.ly/unlock")
print(res.choice, res.confidence) # SCAM 96.4%

# Intent Routing
res = q.get_intent("Please send our corporate VAT tax invoice for October.")
print(res.category, res.confidence) # BILLING 91.0%
```

---

## 🧠 Architectural Highlights

1. **Canonical Symbol Anchoring:** Maps candidate options into single-token anchors `(A), (B), (C)` to eliminate tokenization length disparities and prefix collisions.
2. **Hybrid Contrastive Margin Loss:** Fine-tuned with a safety margin $\gamma = 2.5$ against rival candidate logits.
3. **Context-Free Prior De-Biasing:** Subtracts template frequency bias ($\gamma_{\text{calib}} = 1.25$) on empty context, eliminating false negatives.
4. **Zero-Overhead INT4 Weight Fusion:** Fuses LoRA delta weights directly into base INT4 weights (`merge_and_unload`), eliminating 576 CUDA kernel dispatches and cutting latency from 664ms to 209ms.

---

## 📄 Citation

```bibtex
@misc{morningstar2026systemq,
  title={System Q: Fast System 1 Decision Architecture from Qwen2.5-3B},
  author={Lucifer Morningstar},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/LuciferMorningstar777/system-q-3b-decision}}
}
```