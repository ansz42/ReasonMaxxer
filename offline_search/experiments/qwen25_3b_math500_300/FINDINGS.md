# Qwen2.5-3B MATH-500 offline-search findings

Date: 2026-08-18  
Model: `unsloth/Qwen2.5-3B-Instruct`  
Pack: 300 MATH-500 items (seed 42), 12 search samples, Unsloth LoRA r=16 / α=32 QKVO  
Machine: Vast / CMP 170HX 64 GB

This is a **regression, not a collapse**. The first train recipe over-updated. Search data is kept; training is retried with a milder optimizer.

## Same-protocol greedy 0-shot (vLLM, boxed chat prompt, MathVerifier)

| Model | GSM8K (n=1319) | MATH-500 (n=500) |
| --- | ---: | ---: |
| Base `unsloth/Qwen2.5-3B-Instruct` | **84.6%** | **61.6%** |
| math-test-maxx (first LoRA, v1) | 82.9% | 53.8% |
| Change | −1.7 pp | −7.8 pp |

Official Qwen2.5-3B-Instruct reports GSM8K **86.7%** (typically 8-shot) and MATH **65.9%**. Those are not this harness. The table above is the only apples-to-apples comparison.

## Why this is not catastrophic

- GSM8K stayed in the 80s. A dead or mis-wired trainer would land closer to 20–40%.
- 1305 / 1319 GSM8K completions still contain `\boxed{}`.
- Search found at least one correct sample on **252 / 300** train problems (pass@12 ≈ 84%).
- Search rollouts: 3600, exact-correct 2108 (58.6%), mean graded reward 0.755.

## First train recipe (v1) — what actually ran

| Knob | Configured | Effective |
| --- | --- | --- |
| learning_rate | `2.0e-4` | `2.0e-4` |
| batch_size × grad_accum | 1 × 4 | 1 × 4 |
| max_grad_norm | 1.0 | 1.0 |
| max_steps | 200 | **1548** (`cover_all_informative` raised the cap to one pass over 1548 informative rows) |
| optimizer updates | — | 1548 / 4 = **387** |
| kl_coef | 0.0 | 0.0 |
| train wall | — | 668 s |

Dataset: 1969 selected rows, 1548 informative (914 pos / 763 neg), 421 zero-advantage dropped.

Loss (signed objective, so negative is allowed):

- first 3: −0.61, −0.86, −0.46
- first-50 mean ≈ −0.10
- last-50 mean ≈ −16
- min / max: −414 / +258
- last 3 included −260

That drift is the smoking gun for over-updating, not a broken `decision_loss` formula.

In-loop eval on the **300 train items** (temp 0.6, n=4): pass@1 56.8%, pass@4 71.3%.

W&B (v1):

- search resume / train: https://wandb.ai/batuhan409/offline-search/runs/969656f0
- merge attempt: https://wandb.ai/batuhan409/offline-search/runs/ratnutvz
- LoRA harness: https://wandb.ai/batuhan409/offline-search/runs/ewlzcw58
- base harness: look for `qwen25-base` in the same project

Hub upload of `Ba2han/math-test-maxx` failed: instance `HF_TOKEN` is **read-only**. Merged 16-bit weights are local at `outputs/qwen25_3b_math500_300/merged/math-test-maxx/`.

## Root cause (MATH drop, not implementation collapse)

1. **Too many steps at a high LR.** `cover_all_informative` turned a 200-step cap into 1548 micro-steps at `2e-4` with no KL.
2. **Batch 1** makes the signed advantage estimator very noisy. Accum 4 only yields an effective batch of 4.
3. **`max_grad_norm: 1.0`** does not tame the huge signed-loss spikes (−414 / +258).
4. **763 negative-advantage traces** at that LR can suppress useful MATH reasoning. GSM8K is more redundant, so it held.
5. Search/eval **generation_batch_size: 64** is a throughput knob. It is not why MATH fell.

The loss implementation is the intended signed, length-normalized decision loss:

`L = -sum(advantage * weight * log p) / (sum(weight) + eps)`

## Retry recipe (v2)

Reuse the existing search jsonl + `train_entropy.parquet`. Do not redo search.

| Knob | v1 | v2 |
| --- | ---: | ---: |
| learning_rate | 2.0e-4 | **2.0e-5** |
| batch_size | 1 | **2** |
| gradient_accumulation_steps | 4 | **4** (effective batch **8**) |
| max_grad_norm | 1.0 | **0.1** (was > 0.3) |

`cover_all_informative` is unchanged. With batch 2, one pass is `ceil(1548/2) = 774` micro-steps (~194 optimizer updates) instead of 387, at 10× lower LR and 10× tighter clip.

After v2 train: `04_eval.py` on the 300-pack, then merge + greedy GSM8K / MATH-500 harness for a second same-protocol table.
