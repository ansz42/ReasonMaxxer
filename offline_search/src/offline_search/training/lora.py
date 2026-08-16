from __future__ import annotations

from typing import Any, Sequence


DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def attach_lora_peft(
    model: Any,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
) -> Any:
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_unsloth_base(
    model_name: str,
    *,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=int(max_seq_length),
        load_in_4bit=bool(load_in_4bit),
        dtype=None,
    )


def load_unsloth_lora(
    model_name: str,
    *,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.0,
    target_modules: Sequence[str] | None = None,
    seed: int = 42,
) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=int(max_seq_length),
        load_in_4bit=bool(load_in_4bit),
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(rank),
        target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(seed),
    )
    return model, tokenizer


def load_unsloth_adapter(model: Any, adapter_path: str) -> Any:
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_path))


def save_lora(model: Any, output_dir: str) -> None:
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
        return
    raise TypeError("Model does not support save_pretrained")
