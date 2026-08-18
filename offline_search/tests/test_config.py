from __future__ import annotations

from offline_search.config import ExperimentConfig


def test_default_search_has_eight_temperature_arms():
    cfg = ExperimentConfig()
    arms = cfg.search.sampling_configs()
    assert len(arms) == 8
    assert arms[0].temperature == 0.35
    assert arms[-1].repetition_penalty == 1.05


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  name: unsloth/Qwen3-1.7B",
                "  backend: unsloth",
                "training:",
                "  lora_rank: 16",
                "  objective: graded_signed",
                "search:",
                "  total_samples_per_problem: 8",
            ]
        ),
        encoding="utf-8",
    )
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.model.name == "unsloth/Qwen3-1.7B"
    assert cfg.training.lora_rank == 16
    assert cfg.search.total_samples_per_problem == 8
    assert cfg.training.objective == "graded_signed"


def test_training_defaults_drop_zero_advantage_and_cover_informative():
    cfg = ExperimentConfig()
    assert cfg.training.drop_zero_advantage is True
    assert cfg.training.cover_all_informative is True
    assert cfg.training.min_abs_advantage == 1e-8
    assert cfg.training.save_steps == 100
    assert cfg.training.learning_rate == 2.0e-5
    assert cfg.training.batch_size == 2
    assert cfg.training.gradient_accumulation_steps == 4
    assert cfg.training.max_grad_norm == 0.1
    assert cfg.training.neg_prob_floor == 1e-4
    assert cfg.entropy.mode == "hard"
    assert cfg.selection.drop_clipped is True
    assert cfg.search.retry_clipped is True


def test_search_vllm_runtime_defaults_are_conservative():
    cfg = ExperimentConfig()
    assert cfg.search.gpu_memory_utilization == 0.5
    assert cfg.search.enforce_eager is False


def test_yaml_can_set_vllm_runtime_knobs(tmp_path):
    path = tmp_path / "search.yaml"
    path.write_text(
        "\n".join(
            [
                "search:",
                "  generation_batch_size: 64",
                "  gpu_memory_utilization: 0.5",
                "  enforce_eager: false",
                "model:",
                "  max_seq_length: 6144",
            ]
        ),
        encoding="utf-8",
    )
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.search.generation_batch_size == 64
    assert cfg.search.gpu_memory_utilization == 0.5
    assert cfg.search.enforce_eager is False
    assert cfg.model.max_seq_length == 6144


def test_wandb_section_optional():
    cfg = ExperimentConfig()
    assert cfg.wandb.enabled is False
    cfg = ExperimentConfig.from_mapping({"wandb": {"enabled": True, "project": "offline-search"}})
    assert cfg.wandb.enabled is True
    assert cfg.wandb.project == "offline-search"
