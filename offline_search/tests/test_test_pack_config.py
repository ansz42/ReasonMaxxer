from __future__ import annotations

from pathlib import Path

from offline_search.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]


def test_qwen3_test_pack_is_unsloth_lora():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "test_pack_qwen3_1p7b.yaml")
    assert cfg.model.name == "unsloth/Qwen3-1.7B"
    assert cfg.model.backend == "unsloth"
    assert cfg.training.backend == "unsloth"
    assert cfg.training.lora_rank == 16
    assert cfg.training.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg.training.objective == "graded_signed"
    assert cfg.training.drop_zero_advantage is True
    assert cfg.training.cover_all_informative is True
    assert len(cfg.search.sampling_configs()) == 8
    problems = ROOT / cfg.problems_file
    assert problems.exists()


def test_qwen25_3b_test_pack_is_unsloth_lora_math500():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "test_pack_qwen25_3b.yaml")
    assert cfg.model.name == "unsloth/Qwen2.5-3B-Instruct"
    assert cfg.training.backend == "unsloth"
    assert cfg.training.lora_rank == 16
    assert cfg.training.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg.training.objective == "graded_signed"
    assert cfg.training.drop_zero_advantage is True
    assert cfg.training.cover_all_informative is True
    assert cfg.training.save_steps == 50
    assert cfg.training.learning_rate == 2.0e-5
    assert cfg.training.batch_size == 2
    assert cfg.training.gradient_accumulation_steps == 4
    assert cfg.training.max_grad_norm == 0.1
    assert cfg.search.backend == "vllm"
    assert cfg.search.total_samples_per_problem == 12
    assert cfg.search.max_tokens == 3000
    assert cfg.model.max_seq_length == 6144
    assert cfg.search.generation_batch_size == 64
    assert cfg.search.gpu_memory_utilization == 0.5
    assert cfg.search.enforce_eager is False
    assert cfg.evaluation.max_tokens == 3000
    assert cfg.evaluation.generation_batch_size == 64
    assert len(cfg.search.sampling_configs()) == 8
    problems = ROOT / cfg.problems_file
    assert problems.exists()
    payload = problems.read_text(encoding="utf-8")
    assert '"n": 300' in payload or '"num_records": 300' in payload


def test_qwen25_1p5b_test_pack_is_qkvo_rank16_full_math500():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "test_pack_qwen25_1p5b.yaml")
    assert cfg.model.name == "unsloth/Qwen2.5-1.5B-Instruct"
    assert cfg.training.backend == "unsloth"
    assert cfg.training.lora_rank == 16
    assert cfg.training.lora_alpha == 32
    assert cfg.training.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert cfg.entropy.mode == "hard"
    assert cfg.training.neg_prob_floor == 1e-4
    assert cfg.search.retry_clipped is True
    assert cfg.selection.drop_clipped is True
    assert cfg.training.learning_rate == 1.0e-5
    assert cfg.training.batch_size == 4
    assert cfg.training.gradient_accumulation_steps == 4
    assert cfg.training.warmup_ratio == 0.05
    assert cfg.training.max_grad_norm == 0.1
    assert cfg.search.backend == "vllm"
    assert cfg.search.total_samples_per_problem == 12
    assert cfg.search.max_tokens == 3000
    assert cfg.model.max_seq_length == 6144
    problems = ROOT / cfg.problems_file
    assert problems.exists()
    payload = problems.read_text(encoding="utf-8")
    assert '"n": 500' in payload or '"num_records": 500' in payload


def test_eval_math_harness_is_greedy_gsm8k_math500():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "eval_math_harness.yaml")
    assert cfg.search.backend == "vllm"
    assert cfg.evaluation.temperature == 0.0
    assert cfg.evaluation.n_samples == 1
    assert cfg.evaluation.pass_k == [1]
    assert cfg.evaluation.max_tokens == 3000
    assert cfg.raw.get("benchmarks") == ["gsm8k", "math500"]


def test_eval_math_harness_1p5b_is_greedy_gsm8k_math500():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "eval_math_harness_1p5b.yaml")
    assert cfg.model.name == "unsloth/Qwen2.5-1.5B-Instruct"
    assert cfg.search.backend == "vllm"
    assert cfg.evaluation.temperature == 0.0
    assert cfg.evaluation.n_samples == 1
    assert cfg.evaluation.pass_k == [1]
    assert cfg.evaluation.max_tokens == 3000
    assert cfg.raw.get("benchmarks") == ["gsm8k", "math500"]
    assert cfg.output_dir == "outputs/qwen25_1p5b_math500_500"


def test_eval_aime24_1p5b_is_avg8():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "eval_aime24_1p5b.yaml")
    assert cfg.model.name == "unsloth/Qwen2.5-1.5B-Instruct"
    assert cfg.search.backend == "vllm"
    assert cfg.evaluation.temperature == 0.6
    assert cfg.evaluation.top_p == 0.95
    assert cfg.evaluation.n_samples == 8
    assert cfg.evaluation.pass_k == [1, 8]
    assert cfg.evaluation.max_tokens == 4096
    assert cfg.raw.get("benchmarks") == ["aime24"]
