## AI Generated Experiment

# Qwen2.5-1.5B MATH-500 results

Closed-loop offline-search + entropy-weighted Unsloth LoRA on **all 500** `HuggingFaceH4/MATH-500["test"]` items, then greedy 0-shot GSM8K / MATH-500.

- Model: `unsloth/Qwen2.5-1.5B-Instruct`
- LoRA: r=64, α=128, q/k/o/v/up/down/gate
- Config: `offline_search/configs/test_pack_qwen25_1p5b.yaml`
- Harness: `offline_search/configs/eval_math_harness_1p5b.yaml`
- Machine: Vast / CMP 170HX 64 GB
- Date: 2026-08-18
- Diagnosis: [`FINDINGS.md`](FINDINGS.md)
- W&B: [offline-search](https://wandb.ai/batuhan409/offline-search)

## Same-protocol greedy 0-shot vs the 1.5B base

vLLM, boxed chat prompt, MathVerifier. Not the official few-shot card, and not the 3B table.

| Model | GSM8K (n=1319) | MATH-500 (n=500) |
| --- | ---: | ---: |
| Base `unsloth/Qwen2.5-1.5B-Instruct` | **71.5%** | **50.4%** |
| v1 final (2e-5, 2×4, 1715) | 51.9% | 29.8% |
| v1 checkpoint-500 | 69.4% | 41.6% |
| v2 final (1e-5, 4×4, 5% warmup, 858) | 67.2% | 39.2% |
| v2 vs 1.5B base | −4.2 pp | −11.2 pp |
| v2 vs v1 final | +15.4 pp | +9.4 pp |

Official Qwen2.5-1.5B-Instruct (different protocol): GSM8K 73.2% / MATH 55.2%. Our 0-shot base is 71.5 / 50.4.

## Pack eval (in-domain MATH-500, temp 0.6, n=4)

| Run | pass@1 | pass@4 |
| --- | ---: | ---: |
| v1 final | 30.4% | 48.0% |
| v2 final | **33.2%** | **51.6%** |

## Search / train

Search was run once (6000 rollouts) and reused for v2.

| | v1 | v2 |
| --- | ---: | ---: |
| rollouts | 6000 | same search |
| mean reward | 0.652 | same search |
| exact-correct | 2515 / 6000 | same search |
| informative rows | 3429 / 3692 | 3429 / 3692 (dataset rebuilt with EOS) |
| lr / batch | 2e-5, 2×4 | **1e-5, 4×4**, 5% warmup |
| micro-steps | 1715 | **858** |
| Adam updates | 429 | **214** (10 warmup) |
| last-50 loss mean | −48.7 | −79.5 |
| loss min / max | −446 / +320 | −326 / +280 |
| train wall | 847 s | 908 s |

v2 recovers most of the v1 collapse but still loses to the 1.5B base. v1 checkpoint-500 is the closest trained point (69.4 / 41.6).

Local writes: `FINDINGS.md` (this folder), harness JSON on the box under `outputs/qwen25_1p5b_math500_500/eval_harness/`.
