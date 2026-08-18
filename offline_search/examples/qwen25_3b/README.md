# Qwen2.5-3B-Instruct test pack (LoRA + MATH-500)

Closed-loop recipe on a harder set than the 8-fixture Qwen3 smoke:

```text
300 random MATH-500 test items
  -> vLLM batched search (8 decoding arms, 12 samples, 3000 tokens)
  -> last-2-line regex grader
  -> per-problem advantages
  -> frozen-model token entropy
  -> Unsloth LoRA
  -> pass@1 / pass@k
```

The model id is `unsloth/Qwen2.5-3B-Instruct`. Search and eval generate with vLLM in batches. Training still attaches a rank-16 QKVO LoRA via Unsloth.

Answer capture reuses the boxed / number regexes, but only inspects the last two non-empty lines of a rollout (plus freer cues such as `Final answer:`, `$...$`, and `7/8`).

## 1. Sample the 300 problems

From `offline_search/`:

```powershell
python scripts\sample_math500.py
```

This draws 300 rows from `HuggingFaceH4/MATH-500["test"]` with seed 42 and writes `fixtures/math500_300.json`. Fields kept: `problem`, `solution`, `answer`, `subject`.

## 2. CPU unit pack

```powershell
python examples\qwen25_3b\run_test_pack.py --mode check
python examples\qwen25_3b\run_test_pack.py --mode unit
python examples\qwen25_3b\run_test_pack.py --mode synthetic
```

`unit` includes a gold-answer recall check: the freer last-2-line extractor is run on each of the 300 official solutions.

## 3. GPU loop (Qwen2.5-3B-Instruct + Unsloth LoRA)

```powershell
pip install -r requirements.txt
python examples\qwen25_3b\run_test_pack.py --mode smoke
```

Or step by step:

```powershell
python scripts\01_search.py --config configs\test_pack_qwen25_3b.yaml
python scripts\02_build_dataset.py --config configs\test_pack_qwen25_3b.yaml
python scripts\03_train.py --config configs\test_pack_qwen25_3b.yaml
python scripts\04_eval.py --config configs\test_pack_qwen25_3b.yaml --label qwen25_lora
```

Budget (see `configs/test_pack_qwen25_3b.yaml`):

| knob | value |
| --- | --- |
| model | `unsloth/Qwen2.5-3B-Instruct` |
| problems | 300 random MATH-500 test items |
| search | vLLM, batch 64, max_model_len 6144, gpu_memory_utilization 0.5, eager off, 8 configs x 1 initial, **12 total / problem**, **3000 tokens** |
| LoRA | r=16, alpha=32, QKVO, dropout 0 |
| train | **preferred:** lr `2e-5`, batch **2×4**, `max_grad_norm` **0.1**, Unsloth, checkpoint every 50; `cover_all_informative` may raise `max_steps` to one pass |
| eval | vLLM batched pass@1 and pass@4, n=4, 3000 tokens |

300 problems x 12 rollouts is a real GPU run, not an 8-problem smoke. Research-scale knobs still live in `configs/search.yaml` (256 rollouts / problem).

Measured same-protocol greedy 0-shot (vLLM + MathVerifier):

| Model | GSM8K | MATH-500 |
| --- | ---: | ---: |
| Base `unsloth/Qwen2.5-3B-Instruct` | 84.6% | 61.6% |
| v1 (`2e-4`, 1×4, clip 1.0) | 82.9% | 53.8% |
| v2 preferred (`2e-5`, 2×4, clip 0.1) | **85.0%** | **62.8%** |

Merged weights: [Ba2han/math-test-maxx](https://huggingface.co/Ba2han/math-test-maxx). Full write-up: [`experiments/qwen25_3b_math500_300/FINDINGS.md`](../../experiments/qwen25_3b_math500_300/FINDINGS.md).

## Outputs

```text
outputs/qwen25_3b_math500_300/
  search/search_results.parquet
  search/accounting.json
  dataset/train_entropy.parquet
  train/adapter/
  train/train_metrics.json
  eval/qwen25_lora.json
  merged/math-test-maxx-lr2e5/
  eval_harness/math-test-maxx-lr2e5_summary.json
```

## 4. Merge LoRA, upload, evaluate GSM8K / MATH-500

After training, merge the latest adapter (`train/adapter`, else the highest `checkpoint-*`) into 16-bit weights named `math-test-maxx` and push to the Hub (needs a **write** HF token):

```powershell
python scripts\05_merge_and_push.py --config configs\test_pack_qwen25_3b.yaml --name math-test-maxx
```

Greedy pass@1 harness (vLLM, same boxed chat prompt + MathVerifier). Official Qwen2.5-3B-Instruct GSM8K is 86.7% 8-shot; this run is 0-shot CoT. On this harness the base scored 84.6% GSM8K / 61.6% MATH-500.

```powershell
python scripts\06_eval_benchmarks.py --config configs\eval_math_harness.yaml --model outputs\qwen25_3b_math500_300\merged\math-test-maxx-lr2e5 --label math-test-maxx-lr2e5
```

MATH-500 here is the full 500-item test set. 300 of those items were in the offline-search train pack.
