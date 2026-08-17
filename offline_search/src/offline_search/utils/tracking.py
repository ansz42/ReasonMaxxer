from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

_RUN: Any = None
SEARCH_LOG_EVERY = 8


def should_log_search_progress(step: int, every: int = SEARCH_LOG_EVERY) -> bool:
    """True on the first rollout and every `every` rollouts after that."""
    n = int(step)
    interval = max(1, int(every))
    return n > 0 and (n == 1 or n % interval == 0)


def summarize_search_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = 0
    tokens = 0
    reward_sum = 0.0
    correct = 0
    for record in records:
        n += 1
        tokens += int(record.get("generated_tokens", 0) or 0)
        reward_sum += float(record.get("reward", 0.0) or 0.0)
        correct += int(bool(record.get("is_correct")))
    return {
        "n": n,
        "tokens": tokens,
        "reward_sum": reward_sum,
        "correct": correct,
        "reward_mean": (reward_sum / n) if n else 0.0,
        "correct_rate": (correct / n) if n else 0.0,
    }


def search_progress_metrics(
    step: int,
    record: Mapping[str, Any],
    *,
    tokens: int,
    reward_sum: float,
    correct: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    n = max(1, int(step))
    payload: dict[str, Any] = {
        "search/step": int(step),
        "search/rollouts": int(step),
        "search/tokens": int(tokens),
        "search/reward": float(record.get("reward", 0.0) or 0.0),
        "search/reward_mean": float(reward_sum) / n,
        "search/is_correct": int(bool(record.get("is_correct"))),
        "search/correct_rate": float(correct) / n,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def gpu_snapshot() -> dict[str, Any]:
    stats: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            stats["gpu/name"] = torch.cuda.get_device_name(0)
            stats["gpu/allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 4)
            stats["gpu/reserved_gb"] = round(torch.cuda.memory_reserved() / (1024**3), 4)
            stats["gpu/max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 4)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        ).strip()
        if out:
            row = out.splitlines()[0]
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 6:
                stats["gpu/util_pct"] = float(parts[0])
                stats["gpu/mem_util_pct"] = float(parts[1])
                stats["gpu/mem_used_mb"] = float(parts[2])
                stats["gpu/mem_total_mb"] = float(parts[3])
                stats["gpu/power_w"] = float(parts[4])
                stats["gpu/temp_c"] = float(parts[5])
    except Exception:
        pass
    return stats


def init_from_config(cfg: Any, stage: str) -> Any:
    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None or not bool(getattr(wandb_cfg, "enabled", False)):
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"wandb import failed ({exc}); continuing without it")
        return None

    kwargs: dict[str, Any] = {
        "project": wandb_cfg.project,
        "name": wandb_cfg.name,
        "group": wandb_cfg.group,
        "job_type": stage,
        "config": getattr(cfg, "raw", None) or {},
        "reinit": True,
    }
    if wandb_cfg.entity:
        kwargs["entity"] = wandb_cfg.entity
    if wandb_cfg.mode:
        kwargs["mode"] = wandb_cfg.mode
    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id:
        kwargs["id"] = run_id
        kwargs["resume"] = os.environ.get("WANDB_RESUME", "allow")

    global _RUN
    _RUN = wandb.init(**kwargs)
    if _RUN is not None and getattr(_RUN, "id", None):
        os.environ.setdefault("WANDB_RUN_ID", str(_RUN.id))
        print(f"wandb run: {getattr(_RUN, 'url', _RUN.id)} stage={stage}")
    return _RUN


def log(data: dict[str, Any], step: int | None = None) -> None:
    if _RUN is None:
        return
    try:
        if step is None:
            _RUN.log(data)
        else:
            _RUN.log(data, step=step)
    except Exception as exc:
        print(f"wandb log failed: {exc}")


def finish() -> None:
    global _RUN
    if _RUN is None:
        return
    try:
        _RUN.finish()
    except Exception:
        pass
    _RUN = None
