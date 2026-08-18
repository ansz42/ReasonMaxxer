## AI Generated Experiment

# Offline search + entropy-weighted LoRA

Practical MVP from `reasonmaxxer_offline_codex_handover.md`:

1. Multi-temperature / top-p offline search
2. Graded scalar rewards
3. Adaptive leftover-budget allocation
4. Per-problem signed advantages
5. Frozen-model token entropy
6. Entropy-weighted LoRA (Unsloth; Qwen3-1.7B smoke, Qwen2.5-3B MATH-500 test)
7. pass@1 / pass@k + token accounting

Search and training share one chat-template prefix: generation renders
`<|im_start|>user ... <|im_start|>assistant` and entropy/training append the
assistant tokens to that same prefix. Do not teacher-force the raw user
string. Training also drops near-zero-advantage rows so a `max_steps` cap
cannot spend most of the run on no-ops, and will extend the cap if needed
to see every informative row once.

This directory is an add-on in a fork of ReasonMaxxer. The paper pipeline under `../reasonmaxxer/` is untouched.

Measured Qwen3-1.7B smoke results: [`experiments/qwen3_1p7b_test_pack/RESULTS.md`](experiments/qwen3_1p7b_test_pack/RESULTS.md).

Measured Qwen2.5-3B MATH-500 (greedy 0-shot, same harness): base GSM8K **84.6%** / MATH-500 **61.6%** → v2 LoRA **85.0% / 62.8%**. Preferred train knobs are lr `2e-5`, batch 2×4, `max_grad_norm` 0.1. Write-up: [`experiments/qwen25_3b_math500_300/FINDINGS.md`](experiments/qwen25_3b_math500_300/FINDINGS.md). Merged model: [Ba2han/math-test-maxx](https://huggingface.co/Ba2han/math-test-maxx).

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

Harder follow-up (300 random `HuggingFaceH4/MATH-500["test"]` items, vLLM batched search, last-2-line regex, Unsloth LoRA):

```powershell
python scripts\sample_math500.py
python examples\qwen25_3b\run_test_pack.py --mode unit
python examples\qwen25_3b\run_test_pack.py --mode smoke
```

- model: `unsloth/Qwen2.5-3B-Instruct`
- trainer: Unsloth LoRA, rank 16, QKVO
- config: `configs/test_pack_qwen25_3b.yaml`
- fixtures: `examples/qwen25_3b/fixtures/math500_300.json`

See `examples/qwen25_3b/README.md`.

Full-split 1.5B follow-up (all 500 MATH-500 items, LoRA r=64 on q/k/o/v/up/down/gate):

```powershell
python scripts\sample_math500.py --n 500 --out examples\qwen25_1p5b\fixtures\math500_500.json
python examples\qwen25_1p5b\run_test_pack.py --mode unit
python examples\qwen25_1p5b\run_test_pack.py --mode smoke
```

- model: `unsloth/Qwen2.5-1.5B-Instruct`
- trainer: Unsloth LoRA, rank 64, α=128, QKVO + up/down/gate
- config: `configs/test_pack_qwen25_1p5b.yaml`
- fixtures: `examples/qwen25_1p5b/fixtures/math500_500.json`

See `examples/qwen25_1p5b/README.md`. MATH-500 eval after this pack is in-domain; GSM8K is held-out.

## Layout

```text
configs/          search / train / eval / test-pack YAML
src/offline_search/
scripts/          01_search  02_build_dataset  03_train  04_eval  sample_math500
tests/            CPU unit suite (no model download)
examples/qwen3_1p7b/
examples/qwen25_3b/
examples/qwen25_1p5b/
```
