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


def test_wandb_section_optional():
    cfg = ExperimentConfig()
    assert cfg.wandb.enabled is False
    cfg = ExperimentConfig.from_mapping({"wandb": {"enabled": True, "project": "offline-search"}})
    assert cfg.wandb.enabled is True
    assert cfg.wandb.project == "offline-search"
