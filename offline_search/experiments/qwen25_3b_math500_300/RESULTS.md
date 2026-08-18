## AI Generated Experiment

# Qwen2.5-3B MATH-500 results

Closed-loop offline-search + entropy-weighted Unsloth LoRA on **300** random `HuggingFaceH4/MATH-500["test"]` items (seed 42), then greedy 0-shot GSM8K / MATH-500.

- Model: `unsloth/Qwen2.5-3B-Instruct`
- LoRA: r=16, α=32, QKVO
- Config: `offline_search/configs/test_pack_qwen25_3b.yaml`
- Machine: Vast / CMP 170HX 64 GB
- Date: 2026-08-18
- Merged Hub model: https://huggingface.co/Ba2han/math-test-maxx
- Full diagnosis: [`FINDINGS.md`](FINDINGS.md)
- W&B: [offline-search](https://wandb.ai/batuhan409/offline-search)

## Same-protocol greedy 0-shot (vLLM, boxed chat, MathVerifier)

| Model | GSM8K (n=1319) | MATH-500 (n=500) |
| --- | ---: | ---: |
| Base `unsloth/Qwen2.5-3B-Instruct` | 84.6% | 61.6% |
| v1 LoRA (`2e-4`, batch 1×4, clip 1.0) | 82.9% | 53.8% |
| **v2 LoRA (`2e-5`, batch 2×4, clip 0.1)** | **85.0%** | **62.8%** |
| v2 vs base | **+0.4 pp** | **+1.2 pp** |

Official Qwen2.5-3B-Instruct reports (typically 8-shot) are GSM8K 86.7% / MATH 65.9%. Those are a different protocol.

## Preferred train knobs

These are now the defaults in `configs/train.yaml` and `TrainingConfig`:

| Knob | Preferred |
| --- | --- |
| learning_rate | `2.0e-5` |
| batch_size × grad_accum | **2 × 4** (effective 8) |
| max_grad_norm | **0.1** |
| LoRA | r=16, α=32, QKVO |
| drop_zero_advantage | true |
| cover_all_informative | true |

v1 over-updated: `cover_all_informative` turned a 200-step cap into 1548 micro-steps at `2e-4` with batch 1 and clip 1.0. Loss min/max was −414 / +258. v2 is one pass of 774 micro-steps (~194 Adam updates) at 10× lower LR and 10× tighter clip.

## 300-item pack eval (temp 0.6, n=4)

| Run | pass@1 | pass@4 |
| --- | ---: | ---: |
| v1 | 56.8% | 71.3% |
| v2 | **61.3%** | **74.0%** |

Search (unchanged across v1/v2): 3600 rollouts, mean graded reward 0.755, exact-correct 2108/3600, at least one correct on 252/300 problems (pass@12 ≈ 84%).
