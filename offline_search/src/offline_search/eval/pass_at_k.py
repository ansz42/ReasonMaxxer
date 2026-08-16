from __future__ import annotations

import math
from typing import Sequence


def _comb(n: int, k: int) -> float:
    if k < 0 or n < 0 or k > n:
        return 0.0
    return float(math.comb(int(n), int(k)))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from n samples with c successes."""
    n_i = int(n)
    c_i = int(c)
    k_i = int(k)
    if n_i <= 0 or k_i <= 0:
        return 0.0
    if c_i <= 0:
        return 0.0
    if k_i > n_i:
        return 1.0 if c_i > 0 else 0.0
    if n_i - c_i < k_i:
        return 1.0
    return 1.0 - (_comb(n_i - c_i, k_i) / _comb(n_i, k_i))


def pass_at_k_from_flags(correct_flags: Sequence[bool], ks: Sequence[int]) -> dict[int, float]:
    flags = [bool(x) for x in correct_flags]
    n = len(flags)
    c = sum(1 for x in flags if x)
    return {int(k): pass_at_k(n, c, int(k)) for k in ks}


def empirical_pass_at_k(correct_flags: Sequence[bool], k: int) -> float:
    flags = [bool(x) for x in correct_flags[: int(k)]]
    if not flags:
        return 0.0
    return 1.0 if any(flags) else 0.0
