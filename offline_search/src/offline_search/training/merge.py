from __future__ import annotations

import os
from pathlib import Path
ADAPTER_MARKERS = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def parse_checkpoint_step(path: str | Path) -> int | None:
    name = Path(path).name
    prefix = "checkpoint-"
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    return int(suffix) if suffix.isdigit() else None


def adapter_looks_complete(path: str | Path) -> bool:
    folder = Path(path)
    if not folder.is_dir():
        return False
    names = {child.name for child in folder.iterdir()}
    if names.intersection(ADAPTER_MARKERS):
        return True
    return any(name.endswith(".safetensors") or name.endswith(".bin") for name in names)


def resolve_latest_adapter(train_dir: str | Path) -> Path:
    """Prefer the final `adapter/` dir, otherwise the highest checkpoint-*."""
    root = Path(train_dir)
    final = root / "adapter"
    if adapter_looks_complete(final):
        return final
    checkpoints: list[tuple[int, Path]] = []
    if root.is_dir():
        for child in root.iterdir():
            step = parse_checkpoint_step(child)
            if step is not None and adapter_looks_complete(child):
                checkpoints.append((step, child))
    if checkpoints:
        checkpoints.sort(key=lambda item: item[0])
        return checkpoints[-1][1]
    raise FileNotFoundError(f"No LoRA adapter or checkpoint found under {root}")


def resolve_hub_repo_id(
    *,
    name: str = "math-test-maxx",
    repo_id: str | None = None,
    username: str | None = None,
) -> str:
    if repo_id:
        return str(repo_id)
    cleaned = str(name).strip().strip("/")
    if "/" in cleaned:
        return cleaned
    user = username or os.environ.get("HF_USERNAME") or os.environ.get("HF_USER")
    if not user:
        try:
            from huggingface_hub import whoami

            info = whoami()
            user = info.get("name") if isinstance(info, dict) else None
        except Exception:
            user = None
    if not user:
        raise ValueError("Need --repo-id USER/math-test-maxx or a logged-in Hugging Face user")
    return f"{user}/{cleaned}"


def write_model_card(
    output_dir: str | Path,
    *,
    repo_id: str,
    base_model: str,
    adapter_path: str | Path,
    extra: str = "",
) -> Path:
    dest = Path(output_dir) / "README.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = f"""---
base_model: {base_model}
library_name: transformers
tags:
  - qwen2.5
  - lora
  - math
  - reasonmaxxer
---

# {repo_id.split("/")[-1]}

Merged 16-bit weights: `{base_model}` + LoRA adapter `{adapter_path}`.

Trained with the ReasonMaxxer offline-search loop (entropy-weighted signed LoRA) on 300 MATH-500 items.

{extra}
"""
    dest.write_text(body, encoding="utf-8")
    return dest


def merge_adapter(
    *,
    base_model: str,
    adapter_path: str | Path,
    output_dir: str | Path,
    max_seq_length: int = 6144,
    load_in_4bit: bool = True,
) -> Path:
    """Merge LoRA into the base model and write a full 16-bit Transformers dir."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adapter = Path(adapter_path)
    try:
        return _merge_with_unsloth(
            base_model=base_model,
            adapter_path=adapter,
            output_dir=output,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    except Exception as exc:
        print(f"unsloth merge unavailable ({type(exc).__name__}: {exc}); trying PEFT bf16 merge")
        return _merge_with_peft(
            base_model=base_model,
            adapter_path=adapter,
            output_dir=output,
        )


def _merge_with_unsloth(
    *,
    base_model: str,
    adapter_path: Path,
    output_dir: Path,
    max_seq_length: int,
    load_in_4bit: bool,
) -> Path:
    from peft import PeftModel
    from unsloth import FastLanguageModel

    # Loading the adapter dir lets Unsloth recover base + LoRA from adapter_config.
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=int(max_seq_length),
            load_in_4bit=bool(load_in_4bit),
            dtype=None,
        )
    except Exception:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=int(max_seq_length),
            load_in_4bit=bool(load_in_4bit),
            dtype=None,
        )
        if hasattr(model, "load_adapter"):
            model.load_adapter(str(adapter_path))
        else:
            model = PeftModel.from_pretrained(model, str(adapter_path))
    if hasattr(model, "save_pretrained_merged"):
        model.save_pretrained_merged(str(output_dir), tokenizer, save_method="merged_16bit")
    else:
        merged = model.merge_and_unload()
        merged.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    return output_dir


def _merge_with_peft(*, base_model: str, adapter_path: Path, output_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(model, str(adapter_path)).merge_and_unload()
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def push_model_dir(
    local_dir: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload merged math-test-maxx weights",
) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private, repo_type="model")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"
