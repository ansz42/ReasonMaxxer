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
| Final LoRA (step 1715) | 51.9% | 29.8% |
| checkpoint-500 | 69.4% | 41.6% |
| final vs 1.5B base | −19.6 pp | −20.6 pp |
| ckpt-500 vs 1.5B base | −2.1 pp | −8.8 pp |

Official Qwen2.5-1.5B-Instruct (different protocol): GSM8K 73.2% / MATH 55.2%. Our 0-shot base is 71.5 / 50.4.

## Pack eval (in-domain MATH-500, temp 0.6, n=4)

Trained adapter: pass@1 **30.4%**, pass@4 **48.0%**. Matches the greedy MATH-500 drop on the final merge.

## Search / train

| | |
| --- | ---: |
| rollouts | 6000 |
| mean reward | 0.652 |
| exact-correct | 2515 / 6000 |
| informative rows | 3429 / 3692 |
| train steps | 1715 |
| last-50 loss mean | −48.7 |
| loss min / max | −446 / +320 |
