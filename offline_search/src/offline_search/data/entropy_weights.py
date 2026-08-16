from __future__ import annotations

from typing import Sequence

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def entropy_weights(
    entropies: Sequence[float],
    *,
    threshold: float,
    scale: float,
    mode: str = "sigmoid",
) -> list[float]:
    values = np.asarray(list(entropies), dtype=np.float64)
    if values.size == 0:
        return []
    mode_l = (mode or "sigmoid").strip().lower()
    if mode_l == "hard":
        return (values > float(threshold)).astype(np.float64).tolist()
    if mode_l != "sigmoid":
        raise ValueError(f"Unsupported entropy weight mode: {mode}")
    scale_v = float(scale) if float(scale) != 0.0 else 1e-8
    return sigmoid((values - float(threshold)) / scale_v).tolist()
