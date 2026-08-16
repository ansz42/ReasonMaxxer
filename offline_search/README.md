## AI Generated Experiment

# Offline search + entropy-weighted LoRA

Practical MVP from `reasonmaxxer_offline_codex_handover.md`:

1. Multi-temperature / top-p offline search
2. Graded scalar rewards
3. Adaptive leftover-budget allocation
4. Per-problem signed advantages
5. Frozen-model token entropy
6. Entropy-weighted LoRA (Unsloth on Qwen3-1.7B for the test pack)
7. pass@1 / pass@k + token accounting

This directory is an add-on in a fork of ReasonMaxxer. The paper pipeline under `../reasonmaxxer/` is untouched.

Measured Qwen3-1.7B smoke results: [`experiments/qwen3_1p7b_test_pack/RESULTS.md`](experiments/qwen3_1p7b_test_pack/RESULTS.md).

## Test pack (ready now)

A `.venv` in this directory is for **unit + synthetic** tests. It has numpy/pytest/pyarrow only. A working GPU torch + Unsloth install is required for `--mode smoke`; do not drop a broken Windows CPU torch wheel into this venv.

```powershell
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe examples\qwen3_1p7b\run_test_pack.py --mode check
.\.venv\Scripts\python.exe examples\qwen3_1p7b\run_test_pack.py --mode synthetic
```

The GPU example is pre-wired:

- model: `unsloth/Qwen3-1.7B`
- trainer: Unsloth LoRA, rank 16, QKVO
- config: `configs/test_pack_qwen3_1p7b.yaml`
- fixtures: `examples/qwen3_1p7b/fixtures/smoke_problems.json`

```powershell
python examples\qwen3_1p7b\run_test_pack.py --mode smoke
```

See `examples/qwen3_1p7b/README.md`.

## Layout

```text
configs/          search / train / eval / Qwen3 test-pack YAML
src/offline_search/
scripts/          01_search  02_build_dataset  03_train  04_eval
tests/            CPU unit suite (no model download)
examples/qwen3_1p7b/
```
