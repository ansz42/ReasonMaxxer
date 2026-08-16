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
