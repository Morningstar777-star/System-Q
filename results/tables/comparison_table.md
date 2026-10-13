| Model Architecture | Type | Accuracy | Median Lat (ms) | P95 Lat (ms) | Peak VRAM (GB) | Energy (mJ/dec) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen_Baseline_Generate | Local (RTX 3050) | 50.0% | 402.74 | 457.58 | 1.939 | 12408.94 | Fully Evaluated |
| Qwen_Baseline_DirectScore | Local (RTX 3050) | 75.0% | 132.31 | 166.8 | 1.955 | 5617.6 | Fully Evaluated |
| System_Q_DecisionHead | Local (RTX 3050) | 37.5% | 100.97 | 116.57 | 1.963 | 6194.54 | OK |
| System_Q_Quantized_4Bit | Local (RTX 3050) | 37.5% | 100.97 | 116.57 | 1.963 | 6194.54 | OK |
| System_Q_EarlyExit | Local (RTX 3050) | 50.0% | 105.09 | 113.05 | 1.996 | 4447.41 | Fully Evaluated |
| Laya_Local | Local (RTX 3050) | 75.0% | 182.28 | 231.36 | 4.232 | 5596.43 | Fully Evaluated |
| TypeSafe_Jev_Remote | Remote | 0.0% | N/A | N/A | 3.547 | 100.82 | TYPESAFE_API_KEY not config... |