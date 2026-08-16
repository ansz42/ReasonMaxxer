from __future__ import annotations

from typing import Sequence

import numpy as np


def config_score(correct_rate: float, near_correct_rate: float, mean_reward: float) -> float:
    return 4.0 * float(correct_rate) + 1.0 * float(near_correct_rate) + 0.2 * float(mean_reward)


def softmax_allocation_probs(scores: Sequence[float], temperature: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values
    temp = max(float(temperature), 1e-8)
    shifted = (values / temp) - np.max(values / temp)
    exp = np.exp(shifted)
    total = float(exp.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return exp / total


def hamilton_apportion(weights: Sequence[float], seats: int) -> np.ndarray:
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if seats <= 0:
        return np.zeros(n, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(w)) or float(w.sum()) <= 0.0:
        w = np.ones(n, dtype=np.float64)
    w = w / float(w.sum())
    raw = w * int(seats)
    floors = np.floor(raw).astype(np.int64)
    leftover = int(seats) - int(floors.sum())
    remainders = raw - floors
    order = np.argsort(-remainders, kind="stable")
    for i in range(leftover):
        floors[int(order[i])] += 1
    return floors


def allocate_remaining(
    scores: Sequence[float],
    remaining_budget: int,
    *,
    exploration_fraction: float = 0.20,
    allocation_temperature: float = 0.5,
) -> list[int]:
    """Split leftover budget: (1 - eps) softmax exploit + eps uniform explore."""
    n = len(scores)
    if n == 0:
        return []
    remaining = int(remaining_budget)
    if remaining <= 0:
        return [0] * n

    explore_frac = min(max(float(exploration_fraction), 0.0), 1.0)
    explore_n = int(round(remaining * explore_frac))
    exploit_n = remaining - explore_n

    exploit_probs = softmax_allocation_probs(scores, allocation_temperature)
    explore_probs = np.full(n, 1.0 / n, dtype=np.float64)
    exploit_alloc = hamilton_apportion(exploit_probs, exploit_n)
    explore_alloc = hamilton_apportion(explore_probs, explore_n)
    alloc = exploit_alloc + explore_alloc

    delta = remaining - int(alloc.sum())
    if delta != 0:
        # Repair rare rounding mismatch; prefer the current best config.
        best = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        alloc[best] = max(0, int(alloc[best]) + delta)
    return [int(x) for x in alloc.tolist()]
