# Qwen3-1.7B test pack (LoRA + Unsloth)

Closed-loop recipe for the adaptive offline-search MVP:

```text
8 decoding arms  ->  graded math rewards  ->  per-problem advantages
                 ->  frozen-model token entropy  ->  Unsloth LoRA
                 ->  pass@1 / pass@k
```

The model id is `unsloth/Qwen3-1.7B`. Training attaches a rank-16 QKVO LoRA via Unsloth. Thinking mode is off (`enable_thinking: false`) so the smoke run stays cheap.

## 1. CPU unit pack (this `.venv`)

From `offline_search/`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe examples\qwen3_1p7b\run_test_pack.py --mode check
.\.venv\Scripts\python.exe examples\qwen3_1p7b\run_test_pack.py --mode unit
.\.venv\Scripts\python.exe examples\qwen3_1p7b\run_test_pack.py --mode synthetic
```

`synthetic` runs the same pipeline with a scripted generator and a tiny LM. It does not download Qwen3.

## 2. GPU smoke pack (Qwen3-1.7B + Unsloth LoRA)

Create a CUDA environment, then:

```powershell
pip install -r requirements.txt
python examples\qwen3_1p7b\run_test_pack.py --mode smoke
```

Or step by step:

```powershell
python scripts\01_search.py --config configs\test_pack_qwen3_1p7b.yaml
python scripts\02_build_dataset.py --config configs\test_pack_qwen3_1p7b.yaml
python scripts\03_train.py --config configs\test_pack_qwen3_1p7b.yaml
python scripts\04_eval.py --config configs\test_pack_qwen3_1p7b.yaml --label qwen3_lora
```

Smoke budget (see `configs/test_pack_qwen3_1p7b.yaml`):

| knob | value |
| --- | --- |
| model | `unsloth/Qwen3-1.7B` 4-bit |
| problems | 8 fixtures in `fixtures/smoke_problems.json` |
| search | 8 configs x 2 initial, 8 total / problem, 256 tokens |
| LoRA | r=16, alpha=32, QKVO, dropout 0 |
| train | 20 steps, lr `2e-4`, Unsloth |
| eval | pass@1 and pass@4, n=4 |

Research-scale knobs live in `configs/search.yaml` and `configs/train.yaml` (256 rollouts / problem, 1 epoch).

## Outputs

```text
outputs/qwen3_1p7b_test_pack/
  search/search_results.parquet
  search/accounting.json
  dataset/train_entropy.parquet
  train/adapter/
  train/train_metrics.json
  eval/qwen3_lora.json
```
