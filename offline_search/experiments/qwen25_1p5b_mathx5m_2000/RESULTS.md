## AI Generated Experiment

# Qwen2.5-1.5B MathX-5M results

Offline search + entropy-weighted Unsloth LoRA on **2000** streamed `Modotte/MathX-5M` problems (10 answers each = 20k), then the same greedy 0-shot GSM8K / MATH-500 harness as the MATH-500 1.5B series.

- Model: `unsloth/Qwen2.5-1.5B-Instruct`
- LoRA: r=16, α=32, q/k/o/v/up/down/gate
- Config: `offline_search/configs/test_pack_qwen25_1p5b_mathx5m.yaml`
- Harness: `offline_search/configs/eval_math_harness_1p5b.yaml`
- Machine: Vast / CMP 170HX 64 GB
- Date: 2026-08-18
- Diagnosis: [`FINDINGS.md`](FINDINGS.md)
- W&B: [offline-search](https://wandb.ai/batuhan409/offline-search)

## Same-protocol greedy 0-shot vs the 1.5B base and MATH-500 v3

vLLM, boxed chat prompt, MathVerifier. Not the official few-shot card.

| Model | GSM8K (n=1319) | MATH-500 (n=500) |
| --- | ---: | ---: |
| Base `unsloth/Qwen2.5-1.5B-Instruct` | 71.5% (943) | 50.4% (252) |
| MATH-500 v3 LoRA | **74.3%** (980) | 51.8% (259) |
| **MathX-5M LoRA** | 73.7% (972) | **53.8%** (269) |
| MathX vs 1.5B base | **+2.2 pp** | **+3.4 pp** |
| MathX vs MATH-500 v3 | −0.6 pp | **+2.0 pp** |

MATH-500 is held-out here (training data is streamed MathX-5M, not the MATH-500 test split). MathX is the best MATH-500 score on this 1.5B harness. v3 remains slightly better on GSM8K.

## Search / train

| | MathX-5M |
| --- | ---: |
| problems | 2000 |
| rollouts | 20 000 (10 / problem, single arm, vLLM `n=10`) |
| search wall | 2179 s (~7.4k tok/s) |
| generated tokens | 16.17M |
| exact-correct | 2146 / 20 000 (10.7%) |
| mean reward | 0.473 |
| rows in / informative | 13 265 / 5511 |
| LoRA | r=16 + MLP |
| entropy | hard 0.8 / 0.25 |
| lr / batch | 1e-5, 2×8, 5% warmup |
| micro-steps / updates | 2756 / 344 (17 warmup) |
| train wall | 1117 s |
| train tokens | 4.05M |

Local writes: this folder, harness JSON on the box under `outputs/qwen25_1p5b_math500_500/eval_harness/comparison_mathx.json`. Merged 16-bit: `outputs/qwen25_1p5b_mathx5m_2000/merged/math-test-maxx-1p5b-mathx/`.
