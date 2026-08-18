from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from offline_search.search.sampling_configs import DEFAULT_SEARCH_CONFIG_SPECS, SamplingConfig, configs_from_specs


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return payload


@dataclass
class ModelConfig:
    name: str = "unsloth/Qwen3-1.7B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    enable_thinking: bool = False
    prompt_style: str = "qwen3_chat"
    backend: str = "unsloth"


@dataclass
class SearchConfig:
    initial_samples_per_config: int = 16
    total_samples_per_problem: int = 256
    exploration_fraction: float = 0.20
    allocation_temperature: float = 0.5
    max_tokens: int = 1024
    seed: int = 42
    backend: str | None = None
    generation_batch_size: int = 32
    gpu_memory_utilization: float = 0.5
    enforce_eager: bool = False
    retry_clipped: bool = True
    configs: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_SEARCH_CONFIG_SPECS))

    def sampling_configs(self) -> list[SamplingConfig]:
        return configs_from_specs(self.configs)


@dataclass
class SelectionConfig:
    max_correct_per_problem: int = 8
    max_near_correct_per_problem: int = 16
    max_hard_negatives_per_problem: int = 32
    max_low_reward_negatives_per_problem: int = 4
    drop_clipped: bool = True
    max_generated_tokens: int | None = None


@dataclass
class EntropyConfig:
    threshold: float = 0.8
    scale: float = 0.25
    mode: str = "hard"


@dataclass
class TrainingConfig:
    backend: str = "unsloth"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    learning_rate: float = 2.0e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    max_steps: int | None = None
    kl_coef: float = 0.0
    objective: str = "graded_signed"
    max_grad_norm: float = 0.1
    weight_decay: float = 0.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    seed: int = 42
    logging_steps: int = 5
    save_steps: int | None = 100
    drop_zero_advantage: bool = True
    min_abs_advantage: float = 1e-8
    cover_all_informative: bool = True
    neg_prob_floor: float = 1e-4


@dataclass
class EvaluationConfig:
    pass_k: list[int] = field(default_factory=lambda: [1, 4, 16])
    temperature: float = 0.6
    top_p: float = 0.95
    n_samples: int = 16
    max_tokens: int = 1024
    seed: int = 42
    generation_batch_size: int | None = None


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "offline-search"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    mode: str = "online"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    entropy: EntropyConfig = field(default_factory=EntropyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    problems_file: str | None = None
    output_dir: str = "outputs/run"
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        def section(name: str, typ):
            data = payload.get(name, {}) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Config section {name} must be a mapping")
            allowed = {k: data[k] for k in typ.__dataclass_fields__ if k in data}
            return typ(**allowed)

        return cls(
            model=section("model", ModelConfig),
            search=section("search", SearchConfig),
            selection=section("selection", SelectionConfig),
            entropy=section("entropy", EntropyConfig),
            training=section("training", TrainingConfig),
            evaluation=section("evaluation", EvaluationConfig),
            wandb=section("wandb", WandbConfig),
            problems_file=payload.get("problems_file"),
            output_dir=str(payload.get("output_dir", "outputs/run")),
            raw=payload,
        )

    @classmethod
    def from_yaml(cls, *paths: str | Path) -> "ExperimentConfig":
        merged: dict[str, Any] = {}
        for path in paths:
            merged = _deep_merge(merged, load_yaml(path))
        return cls.from_mapping(merged)
