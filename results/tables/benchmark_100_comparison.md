# Head-to-Head 100-Question Evaluation Benchmark

**Hardware Environment:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM), Windows 11, PyTorch 2.5.1+cu121

| Model Configuration | Architecture | Overall Acc (100 Qs) | Scam Detection (50 Qs) | Email Intent (50 Qs) | Median Latency | Peak VRAM | Energy / Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Laya (ModernBERT-large RL Agent)** | Local (Encoder 421M) | **67.0%** | 82.0% | 52.0% | 43.93 ms | 2.262 GB | 2097 mJ |
| **System Q (Zero-Shot Direct Scorer)** | Local (Qwen2.5-3B INT4) | **38.0%** | 50.0% | 26.0% | 496.94 ms | 1.962 GB | 17944 mJ |
| **System Q v2 (Unmerged LoRA)** | Local (INT4 + Dynamic LoRA) | **56.0%** | 50.0% | 62.0% | 664.76 ms | 2.006 GB | 25248 mJ |
| **System Q SOTA (Fused INT4 + Margin LoRA)** | Local (INT4 Fused + Prior De-Biasing) | **88.0%** | 88.0% | 88.0% | 209.97 ms | 2.003 GB | 14156 mJ |

### Key Takeaways
1. **SOTA Superiority:** System Q SOTA achieves **88.0% Overall Accuracy**, exceeding Laya (67.0%) by **+21.0 percentage points**.
2. **Intent Accuracy Dominance:** System Q reaches **88.0%** on Intent Classification compared to Laya's **52.0%** (+36.0% gap).
3. **Scam Accuracy Superiority:** System Q reaches **88.0%** on Scam Detection compared to Laya's **82.0%** (+6.0% gain).
4. **Zero Generation Tokens:** Forward pass evaluates canonical token logits in a single inference step with 0 autoregressive generation overhead.
5. **Fused LoRA Latency:** Weight fusion (`merge_and_unload`) drops median latency to **~209 ms**, down 3.16x from unmerged LoRA (664 ms).
6. **Memory Efficiency:** System Q peak VRAM is **2.003 GB**, well within the 4.0 GB RTX 3050 hardware budget, and lower than Laya (2.262 GB).
