from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def per_problem_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    """Normalize rewards within one problem: (r - mean) / (std + eps)."""
    values = np.asarray(list(rewards), dtype=np.float64)
    if values.size == 0:
        return []
    centered = values - float(values.mean())
    std = float(values.std())
    return ((centered) / (std + float(eps))).tolist()


def attach_advantages(
    records: Sequence[dict[str, Any]],
    *,
    reward_key: str = "reward",
    eps: float = 1e-6,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        grouped[str(record.get("problem_id", ""))].append(i)

    out = [dict(record) for record in records]
    for indices in grouped.values():
        rewards = [float(out[i].get(reward_key, 0.0)) for i in indices]
        advantages = per_problem_advantages(rewards, eps=eps)
        for i, advantage in zip(indices, advantages):
            out[i]["advantage"] = float(advantage)
    return out
