# Qwen2.5-1.5B MathX-5M offline-search findings

Date: 2026-08-18  
Model: `unsloth/Qwen2.5-1.5B-Instruct`  
Pack: **2000** streamed `Modotte/MathX-5M` items (`problem`, `expected_answer`), **10** vLLM answers each (**20k**).  
LoRA: r=16 / α=32 on q/k/o/v/up/down/gate, same v3 signed-entropy recipe (hard mask, drop clipped, `neg_prob_floor=1e-4`).  
Machine: Vast / CMP 170HX 64 GB

Comparison is against the **same 1.5B base** and the MATH-500-pack v3 adapter, same greedy 0-shot harness. Official Qwen2.5-1.5B-Instruct reports GSM8K **73.2%** / MATH **55.2%** under a different (typically few-shot) protocol.

## Same-protocol greedy 0-shot (vLLM, boxed chat prompt, MathVerifier)

| Model | GSM8K (n=1319) | MATH-500 (n=500) |
| --- | ---: | ---: |
| Base `unsloth/Qwen2.5-1.5B-Instruct` | 71.5% (943) | 50.4% (252) |
| MATH-500 v3 LoRA | **74.3%** (980) | 51.8% (259) |
| **MathX-5M LoRA** | 73.7% (972) | **53.8%** (269) |
| MathX vs 1.5B base | **+2.2 pp** | **+3.4 pp** |
| MathX vs MATH-500 v3 | −0.6 pp | **+2.0 pp** |

GSM8K and MATH-500 are both held-out relative to MathX-5M (the 2000 streamed rows are not the MATH-500 test split).

MathX is the **best MATH-500** number on this harness so far (+10 items vs v3, +17 vs base). v3 still leads GSM8K by 8 items. That split is plausible: v3 distilled MATH-500-style contest traces; MathX distilled 2000 more diverse streamed problems and transferred better to MATH-500 than to grade-school word problems.

## What ran

| Stage | Result |
| --- | --- |
| sample | streamed first 2000 usable `Modotte/MathX-5M` rows; parquet columns `problem`, `expected_answer`; 0 empty / 0 over-long skipped |
| generate | 20 000 rollouts, one vLLM pass `n=10` temp 0.85 / top_p 0.95, batch 16 prompts (160 sequences), 36.3 min, 16.17M tokens (~7.4k tok/s) |
| search score | exact-correct 2146 / 20 000 (**10.7%**), mean reward 0.473 |
| dataset | 13 265 rows / **5511** informative / 9575 pos / 3562 neg / 7754 zero-advantage dropped |
| train | 2756 micro-steps (batch 2 × accum 8), 344 Adam updates (17 warmup), 1117 s, 4.05M tokens |
| loss (signed) | first −0.250 → last −0.848, mean −0.004 |

Search used a **flat parallel** generator (`scripts/generate_mathx_parallel.py`), not the 8-arm adaptive allocator. All 20k jobs share one sampling arm so vLLM can prefix-share `n=10`.

Train knobs match v3 except memory: batch **2×8** (effective 16) instead of 4×4, `max_seq_length` 6144, `max_tokens` 2500. `cover_all_informative` raised the 400-step cap to one pass over 5511 rows.

A transient CUDA allocator warning fired during entropy (same class as v3). The dataset write finished. vLLM logged the usual Numba/NumPy 2.5 import warning; it did not stop generation.

## How this differs from the MATH-500 pack

| | MATH-500 v3 | MathX-5M |
| --- | --- | --- |
| prompts | 500 MATH-500 **test** items (in-domain for the MATH-500 bench) | 2000 streamed MathX-5M (held-out vs both benches) |
| rollouts | 6000 (12 / problem, 8-arm adaptive, reused for v1–v3) | **20 000** (10 / problem, single arm, parallel) |
| exact-correct at search | 41.9% | **10.7%** (harder / more diverse) |
| informative train rows | 2968 | **5511** |
| MATH-500 greedy | 51.8% | **53.8%** |
| GSM8K greedy | **74.3%** | 73.7% |

The 10.7% search hit rate is the regime the handover asked for (rare successes). Offline distillation still moved pass@1 on MATH-500 by +3.4 pp over the base without training on that split.

## W&B / artifacts

- search / dataset / train: https://wandb.ai/batuhan409/offline-search/runs/28f0e810
- merge: https://wandb.ai/batuhan409/offline-search/runs/bea3eded
- harness: https://wandb.ai/batuhan409/offline-search/runs/6976a7ff

On the box:

- parquet: `data/mathx5m/mathx5m_2000.parquet`
- problems JSON: `examples/qwen25_1p5b/fixtures/mathx5m_2000.json`
- search: `outputs/qwen25_1p5b_mathx5m_2000/search/search_results.jsonl`
- adapter: `outputs/qwen25_1p5b_mathx5m_2000/train/adapter/`
- merged 16-bit: `outputs/qwen25_1p5b_mathx5m_2000/merged/math-test-maxx-1p5b-mathx/`
- comparison: `outputs/qwen25_1p5b_math500_500/eval_harness/comparison_mathx.json`

Not pushed to the Hub (`--no-push`). No AIME24 run for this adapter yet.
