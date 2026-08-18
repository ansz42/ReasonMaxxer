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
| Base `unsloth/Qwen2.5-1.5B-Instruct` | 71.5% (943) | 50.4% (252) |
| v1 final (2e-5, 2×4, 1715) | 51.9% (684) | 29.8% (149) |
| v1 checkpoint-500 | 69.4% (915) | 41.6% (208) |
| v2 final (1e-5, 4×4, 5% warmup, 858) | 67.2% (887) | 39.2% (196) |
| **v3 final** (1e-5, 4×4, r=16, hard mask) | **74.3%** (980) | **51.8%** (259) |
| v3 vs 1.5B base | **+2.8 pp** | **+1.4 pp** |
| v3 vs v2 | +7.1 pp | +12.6 pp |

Official Qwen2.5-1.5B-Instruct (different protocol): GSM8K 73.2% / MATH 55.2%. Our 0-shot base is 71.5 / 50.4. v3 is the first adapter on this pack that beats the 1.5B base on both benches.

## Pack eval (in-domain MATH-500, temp 0.6, n=4)

| Run | pass@1 | pass@4 |
| --- | ---: | ---: |
| v1 final | 30.4% | 48.0% |
| v2 final | 33.2% | 51.6% |
| **v3 final** | **51.4%** | **66.8%** |

## Search / train

Search was run once (6000 rollouts) and reused for v2 and v3.

| | v1 | v2 | v3 |
| --- | ---: | ---: | ---: |
| rollouts | 6000 | same search | same search |
| mean reward | 0.652 | same search | same search |
| exact-correct | 2515 / 6000 | same search | same search |
| rows in / informative | 3692 / 3429 | 3692 / 3429 (EOS rebuild) | **3538 / 2968** (drop clipped + zero-A) |
| LoRA | r=64 + MLP | r=64 + MLP | **r=16 + MLP** |
| entropy | soft sigmoid | soft sigmoid | **hard 0.8 / 0.25** |
| lr / batch | 2e-5, 2×4 | 1e-5, 4×4, 5% warmup | same as v2 |
| micro-steps | 1715 | 858 | **742** |
| Adam updates | 429 | 214 (10 warmup) | **185** (9 warmup) |
| last-50 loss mean | −48.7 | −79.5 | **−0.51** |
| loss min / max | −446 / +320 | −326 / +280 | **−6.22 / +1.93** |
| train wall | 847 s | 908 s | **493 s** |

v3 is the first 1.5B point that beats the base. Loss stays bounded after the per-sequence mean, hard entropy mask, clipped-trace drop, and `neg_prob_floor`.

Local writes: `FINDINGS.md` (this folder), harness JSON on the box under `outputs/qwen25_1p5b_math500_500/eval_harness/` (`comparison_v3.json`).
