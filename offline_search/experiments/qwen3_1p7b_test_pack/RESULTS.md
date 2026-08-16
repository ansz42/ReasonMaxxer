## AI Generated Experiment

# Qwen3-1.7B test-pack results

Closed-loop smoke on **NVIDIA L40S** (47.4 GB, CUDA 13.0, torch `2.11.0+cu130`, Unsloth `2026.8.18`).

- Model: `unsloth/Qwen3-1.7B` 4-bit
- LoRA: r=16, α=32, targets `q_proj k_proj v_proj o_proj`
- Config: `offline_search/configs/test_pack_qwen3_1p7b.yaml`
- Problems: 8 fixtures in `examples/qwen3_1p7b/fixtures/smoke_problems.json`
- Date: 2026-08-16
- W&B project: https://wandb.ai/batuhan409/offline-search

| Stage | W&B run |
| --- | --- |
| Search | https://wandb.ai/batuhan409/offline-search/runs/i0x6yioq |
| Dataset | https://wandb.ai/batuhan409/offline-search/runs/uofpgqhj |
| Train | https://wandb.ai/batuhan409/offline-search/runs/hw0cgkfo |
| Eval | https://wandb.ai/batuhan409/offline-search/runs/xh8md6df |

Adapter weights and raw parquet were left on the ephemeral Modal volume and are not checked in. Metrics below are copied from the logged JSON of the completed run.

## Wall time

| Stage | Seconds |
| --- | ---: |
| Search | 281.2 |
| Dataset (entropy + select) | ~28 |
| Train (20 steps) | 4.65 |
| Eval (32 samples) | ~70 |
| **Full loop** | **465.5** |

Search generated 13,732 tokens at 48.8 tok/s. GPU-hours ≈ 0.13.

## GPU (L40S, 49140 MiB)

The 1.7B 4-bit pack barely loads the card (sequential `n=1` generate, train batch 1).

| Stage | torch allocated | nvidia-smi used | Util | Power | Temp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Search | 1.40 GB (max 1.43) | 2050 MB | 37% | 121 W | 42 °C |
| Dataset / entropy | 1.34 GB (max 1.99) | 3156 MB | 19% | — | — |
| Train (peak) | 1.54 GB (max 1.84) | 2946 MB | 8–33% | 87–130 W | 37–40 °C |
| Eval | 1.44 GB (max 1.47) | 2086 MB | 9% | — | — |

## Search

- 128 / 128 rollouts (8 problems × 16; 8 arms × 1 initial + 8 leftover)
- **124 / 128 correct (96.9%)**, mean reward **0.981**
- All 4 misses are `word/work-rate`
- Adaptive leftover allocation ran (lower-temp arms received more leftover samples)

A first pass treated `\frac{7}{8}` as incorrect against gold `7/8`. The scorer now converts `\frac` before stripping braces. Re-score of the same rollouts is 124/128; live eval after the fix also treats the fraction item as correct.

## Dataset

| Field | Value |
| --- | ---: |
| rows | 36 |
| problems | 8 |
| positive advantages | 4 |
| negative advantages | 4 |
| mean reward | 0.933 |

Only `word/work-rate` produced mixed rewards, so only that problem has non-zero signed advantages.

## Train losses (20 steps)

Mean loss **0.060**. Zero-advantage easy problems contribute 0. Non-zero steps are the signed `work-rate` rows:

| Step | Loss | Advantage | Tokens |
| ---: | ---: | ---: | ---: |
| 5 | −1.375 | −1.00 | 256 |
| 7 | +1.289 | +1.00 | 256 |
| 11 | +1.258 | +1.00 | 256 |
| 14 | +1.328 | +1.00 | 256 |
| 18 | −1.297 | −1.00 | 256 |
| other 15 steps | 0.000 | 0.00 | 19–142 |

Negative loss = down-weight a hard negative. Positive loss = up-weight a correct trajectory.

## Eval (trained adapter, n=4)

| Metric | Value |
| --- | ---: |
| pass@1 | **0.9375** (30/32) |
| pass@4 | **1.000** |

| Problem | n correct / 4 | pass@1 | pass@4 |
| --- | ---: | ---: | ---: |
| arith/add-17-28 | 4 | 1.00 | 1.00 |
| arith/mul-12-8 | 4 | 1.00 | 1.00 |
| arith/frac-3-4-plus-1-8 | 4 | 1.00 | 1.00 |
| word/apples | 4 | 1.00 | 1.00 |
| word/remainder | 4 | 1.00 | 1.00 |
| word/percent | 4 | 1.00 | 1.00 |
| word/work-rate | 2 | 0.50 | 1.00 |
| word/linear | 4 | 1.00 | 1.00 |

These fixtures are too easy for Qwen3-1.7B. The signed objective only fires on `work-rate`. A harder set is required to test whether rare pass@k becomes better pass@1.
