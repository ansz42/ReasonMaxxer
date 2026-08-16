from __future__ import annotations

import os
import subprocess
from typing import Any

_RUN: Any = None


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
