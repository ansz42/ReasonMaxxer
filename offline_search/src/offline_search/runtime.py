from __future__ import annotations

from pathlib import Path
from typing import Any

from offline_search.config import ExperimentConfig
from offline_search.data.problems import load_problems_file
from offline_search.search.generate import ScriptedBackend, TransformersBackend, UnslothBackend
from offline_search.search.search_runner import SearchSettings
from offline_search.scoring.math_verifier import MathVerifier


def load_experiment(*config_paths: str | Path) -> ExperimentConfig:
    return ExperimentConfig.from_yaml(*config_paths)


def search_settings_from_config(cfg: ExperimentConfig) -> SearchSettings:
    s = cfg.search
    return SearchSettings(
        initial_samples_per_config=s.initial_samples_per_config,
        total_samples_per_problem=s.total_samples_per_problem,
        exploration_fraction=s.exploration_fraction,
        allocation_temperature=s.allocation_temperature,
        max_tokens=s.max_tokens,
        seed=s.seed,
    )


def load_problems(cfg: ExperimentConfig):
    if not cfg.problems_file:
        raise ValueError("problems_file is required")
    return load_problems_file(cfg.problems_file, prompt_style=cfg.model.prompt_style)


def make_scorer() -> MathVerifier:
    return MathVerifier()


def load_generation_stack(cfg: ExperimentConfig, adapter_path: str | Path | None = None) -> tuple[Any, Any, Any]:
    backend_name = (cfg.model.backend or "unsloth").lower()
    if backend_name == "scripted":
        return None, None, ScriptedBackend()
    if backend_name == "unsloth":
        from offline_search.training.lora import load_unsloth_base, load_unsloth_adapter

        model, tokenizer = load_unsloth_base(
            cfg.model.name,
            max_seq_length=cfg.model.max_seq_length,
            load_in_4bit=cfg.model.load_in_4bit,
        )
        adapter = adapter_path or cfg.raw.get("adapter_path")
        if adapter:
            model = load_unsloth_adapter(model, adapter)
        return model, tokenizer, UnslothBackend(model, tokenizer, enable_thinking=cfg.model.enable_thinking)
    if backend_name == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name, trust_remote_code=True)
        return model, tokenizer, TransformersBackend(model, tokenizer, enable_thinking=cfg.model.enable_thinking)
    raise ValueError(f"Unknown model.backend: {cfg.model.backend}")
