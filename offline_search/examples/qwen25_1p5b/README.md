# Qwen2.5-1.5B-Instruct pack (rank-64 LoRA + full MATH-500)

Closed-loop recipe on **all 500** `HuggingFaceH4/MATH-500["test"]` items:

```text
500 MATH-500 test items
  -> vLLM batched search (8 decoding arms, 12 samples, 3000 tokens)
  -> last-2-line regex grader
  -> per-problem advantages
  -> frozen-model token entropy
  -> Unsloth LoRA r=64 on q/k/o/v/up/down/gate
  -> pass@1 / pass@k
```

The model id is `unsloth/Qwen2.5-1.5B-Instruct`. Search and eval generate with vLLM in batches. Training attaches a rank-64 LoRA (α=128) on attention **and** MLP projections.

Because this pack trains on the **full** MATH-500 test split, a later MATH-500 harness is in-domain. GSM8K stays held-out.

## 1. Sample all 500 problems

From `offline_search/`:

```powershell
python scripts\sample_math500.py --n 500 --out examples\qwen25_1p5b\fixtures\math500_500.json
```

## 2. CPU unit pack

```powershell
python examples\qwen25_1p5b\run_test_pack.py --mode check
python examples\qwen25_1p5b\run_test_pack.py --mode unit
python examples\qwen25_1p5b\run_test_pack.py --mode synthetic
```

## 3. GPU loop

```powershell
pip install -r requirements.txt
python examples\qwen25_1p5b\run_test_pack.py --mode smoke
```

Or step by step:

```powershell
python scripts\01_search.py --config configs\test_pack_qwen25_1p5b.yaml
python scripts\02_build_dataset.py --config configs\test_pack_qwen25_1p5b.yaml
python scripts\03_train.py --config configs\test_pack_qwen25_1p5b.yaml
python scripts\04_eval.py --config configs\test_pack_qwen25_1p5b.yaml --label qwen25_1p5b_lora
```

| knob | value |
| --- | --- |
| model | `unsloth/Qwen2.5-1.5B-Instruct` |
| problems | **all 500** MATH-500 test items |
| search | vLLM, batch 64, max_model_len 6144, gpu_memory_utilization 0.5, eager off, 8 configs × 1 initial, **12 total / problem**, **3000 tokens** |
| LoRA | r=64, alpha=128, **q/k/o/v/up/down/gate**, dropout 0 |
| train | lr `1e-5`, batch 4×4, 5% warmup, `max_grad_norm` 0.1, Unsloth, checkpoint every 50 |
| eval | pack: vLLM pass@1/@4, n=4, 3000 tokens; harness: greedy GSM8K + MATH-500 via `06_eval_benchmarks.py` |

500 problems × 12 rollouts = 6000 search samples.

## Outputs

```text
outputs/qwen25_1p5b_math500_500/
  search/search_results.jsonl
  dataset/train_entropy.parquet
  train/adapter/
  train/checkpoint-50 ...
  eval/qwen25_1p5b_lora.json
  eval_harness/qwen25-1p5b-base_summary.json
  eval_harness/qwen25-1p5b-lora_summary.json
```

Greedy pass@1 harness (same boxed chat prompt + MathVerifier as the 3B table). MATH-500 is in-domain for this pack; GSM8K is held-out.

```powershell
python scripts\05_merge_and_push.py --config configs\test_pack_qwen25_1p5b.yaml --name math-test-maxx-1p5b --no-push
python scripts\06_eval_benchmarks.py --config configs\eval_math_harness_1p5b.yaml --model unsloth/Qwen2.5-1.5B-Instruct --label qwen25-1p5b-base
python scripts\06_eval_benchmarks.py --config configs\eval_math_harness_1p5b.yaml --model outputs\qwen25_1p5b_math500_500\merged\math-test-maxx-1p5b --label qwen25-1p5b-lora
```
