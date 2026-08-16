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
| search | vLLM, batch 32, 8 configs x 1 initial, **12 total / problem**, **3000 tokens** |
| LoRA | r=16, alpha=32, QKVO, dropout 0 |
| train | up to 200 steps on **nonzero-advantage** rows, lr `2e-4`, Unsloth |
| eval | vLLM batched pass@1 and pass@4, n=4, 3000 tokens |

300 problems x 12 rollouts is a real GPU run, not an 8-problem smoke. Research-scale knobs still live in `configs/search.yaml` (256 rollouts / problem).

## Outputs

```text
outputs/qwen25_3b_math500_300/
  search/search_results.parquet
  search/accounting.json
  dataset/train_entropy.parquet
  train/adapter/
  train/train_metrics.json
  eval/qwen25_lora.json
```
