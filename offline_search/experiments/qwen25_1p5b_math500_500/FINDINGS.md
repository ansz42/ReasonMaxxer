# Qwen2.5-1.5B MATH-500 offline-search findings

Date: 2026-08-18  
Model: `unsloth/Qwen2.5-1.5B-Instruct`  
Pack: **all 500** MATH-500 test items, 12 search samples, Unsloth LoRA r=64 / α=128 on q/k/o/v/up/down/gate  
Machine: Vast / CMP 170HX 64 GB

Comparison is against the **same 1.5B base**, not the 3B table. Official Qwen2.5-1.5B-Instruct reports GSM8K **73.2%** and MATH **55.2%**; those are a different protocol (typically few-shot). The table below is the only apples-to-apples number.

## Same-protocol greedy 0-shot (vLLM, boxed chat prompt, MathVerifier)

| Model | GSM8K (n=1319) | MATH-500 (n=500) |
| --- | ---: | ---: |
| Base `unsloth/Qwen2.5-1.5B-Instruct` | **71.5%** (943) | **50.4%** (252) |
| Final LoRA (step 1715) | 51.9% (684) | 29.8% (149) |
| checkpoint-500 | 69.4% (915) | 41.6% (208) |
| final vs 1.5B base | **−19.6 pp** | **−20.6 pp** |
| ckpt-500 vs 1.5B base | −2.1 pp | −8.8 pp |

GSM8K is held-out. MATH-500 is in-domain (the pack trains on the full test split).

Our 1.5B base lands close to the official card (71.5 vs 73.2 GSM8K, 50.4 vs 55.2 MATH) once you allow for 0-shot vs few-shot. The trained adapters do **not** beat this base.

## What ran

| Stage | Result |
| --- | --- |
| search | 6000 rollouts, 21.8 min, 4.28M tokens, mean reward 0.652, exact-correct 2515/6000 (41.9%) |
| dataset | 3692 rows / 3429 informative / 1941 pos / 1707 neg / 263 zero-advantage |
| train | 1715 micro-steps (`cover_all_informative` × batch 2), 429 Adam updates, 847 s, 2.71M tokens |
| pack eval (temp 0.6, n=4, 500 train items) | pass@1 **30.4%**, pass@4 **48.0%** |

Train knobs were the preferred 3B-v2 recipe (lr `2e-5`, batch 2×4, clip 0.1) but on a **rank-64 QKVO+MLP** adapter and a **full 500-item** informative set, so the step count is larger (1715 vs 774).

Loss (signed objective):

- first 3: +0.68, −0.04, −14.3
- first-50 mean ≈ −2.67
- last-50 mean ≈ −48.7
- min / max: −446 / +320
- last 3: +52.3, −20.6, +0.36

That range is the smoking gun for over-updating: more LoRA capacity (r=64 + MLP) and more steps (1715) than the 1.5B base can absorb at this objective.

## Format / extraction

`\boxed{}` rate on the greedy harness:

| Model | GSM8K boxed | MATH-500 boxed |
| --- | ---: | ---: |
| 1.5B base | 1163 / 1319 (88.2%) | 460 / 500 (92.0%) |
| checkpoint-500 | 1266 / 1319 (96.0%) | 420 / 500 (84.0%) |
| final LoRA | 932 / 1319 (70.7%) | 260 / 500 (52.0%) |

The final adapter is not dead (GSM8K still 51.9%), but it loses the boxed format on MATH and drops ~20 pp on both benches versus the 1.5B base. Checkpoint-500 keeps GSM8K almost intact (−2.1 pp) and still loses 8.8 pp on MATH.

## W&B

- search / train / pack eval: https://wandb.ai/batuhan409/offline-search/runs/d7a8045c
- base harness: https://wandb.ai/batuhan409/offline-search/runs/835f8260
- final LoRA harness: https://wandb.ai/batuhan409/offline-search/runs/78282032
- checkpoint-500 harness: look for `qwen25-1p5b-ckpt500-harness` in the same project

Local merges (not pushed): `outputs/qwen25_1p5b_math500_500/merged/math-test-maxx-1p5b/` and `.../math-test-maxx-1p5b-ckpt500/`.
