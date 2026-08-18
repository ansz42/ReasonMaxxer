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
| v1 final (2e-5, 2×4, 1715 steps) | 51.9% (684) | 29.8% (149) |
| v1 checkpoint-500 | 69.4% (915) | 41.6% (208) |
| v2 final (1e-5, 4×4, 5% warmup, 858 steps) | 67.2% (887) | 39.2% (196) |
| v1 final vs 1.5B base | −19.6 pp | −20.6 pp |
| v2 final vs 1.5B base | −4.2 pp | −11.2 pp |
| v2 vs v1 final | **+15.4 pp** | **+9.4 pp** |

GSM8K is held-out. MATH-500 is in-domain (the pack trains on the full test split).

Our 1.5B base lands close to the official card (71.5 vs 73.2 GSM8K, 50.4 vs 55.2 MATH) once you allow for 0-shot vs few-shot. The trained adapters do **not** beat this base.

## What ran (v1)

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
| v1 checkpoint-500 | 1266 / 1319 (96.0%) | 420 / 500 (84.0%) |
| v1 final | 932 / 1319 (70.7%) | 260 / 500 (52.0%) |
| v2 final | 1186 / 1319 (89.9%) | 409 / 500 (81.8%) |

The final adapter is not dead (GSM8K still 51.9%), but it loses the boxed format on MATH and drops ~20 pp on both benches versus the 1.5B base. Checkpoint-500 keeps GSM8K almost intact (−2.1 pp) and still loses 8.8 pp on MATH.

## W&B

- search / train / pack eval: https://wandb.ai/batuhan409/offline-search/runs/d7a8045c
- base harness: https://wandb.ai/batuhan409/offline-search/runs/835f8260
- final LoRA harness: https://wandb.ai/batuhan409/offline-search/runs/78282032
- checkpoint-500 harness: look for `qwen25-1p5b-ckpt500-harness` in the same project

Local merges (not pushed): `outputs/qwen25_1p5b_math500_500/merged/math-test-maxx-1p5b/` and `.../math-test-maxx-1p5b-ckpt500/`.

## v2 retry (lr 1e-5, batch 4×4, 5% warmup)

Reused the same 6000-rollout search. Rebuilt `dataset_v2` so training sequences end with `<|im_end|>`. Did not redo search.

Setup audit (this was not the chat-template bug):

- Search prefixes match `tokenizer.apply_chat_template(..., add_generation_prompt=True)` exactly, including the default Qwen system line.
- EOS is `<|im_end|>` (151645). v1 responses did **not** include it; v2 appends it once.
- v1 padded batches with token **0**. v2 pads with the tokenizer pad id (`<|vision_pad|>` / 151654, a reserved unused id).
- Warmup was configured as `warmup_steps: 0` and never applied. v2 uses `warmup_ratio: 0.05` → **10** of **214** Adam updates.

| Knob | v1 | v2 |
| --- | ---: | ---: |
| learning_rate | 2e-5 | **1e-5** |
| batch × accum | 2 × 4 (eff. 8) | **4 × 4** (eff. 16) |
| warmup | none | **5%** (10 updates) |
| micro-steps | 1715 | **858** |
| Adam updates | 429 | **214** |
| train wall | 847 s | 908 s |

Loss is still signed and spiky (last-50 mean −79.5, min/max −326 / +280), but the harness recovered most of the v1 collapse. Boxed format is back near the 1.5B base. v2 still does **not** beat the 1.5B base. v1 checkpoint-500 remains the closest trained point on this harness (69.4 / 41.6).

In-pack eval (temp 0.6, n=4): v1 30.4 / 48.0 → v2 **33.2 / 51.6**.

W&B v2 train: https://wandb.ai/batuhan409/offline-search/runs/2b80d281  
W&B v2 harness: look for `qwen25-1p5b-lora-v2-harness`  
Merged 16-bit: `outputs/qwen25_1p5b_math500_500/merged/math-test-maxx-1p5b-v2/`
