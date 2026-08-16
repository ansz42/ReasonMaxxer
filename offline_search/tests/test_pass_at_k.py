from __future__ import annotations

from offline_search.eval.pass_at_k import pass_at_k, pass_at_k_from_flags


def test_pass_at_1_is_empirical_rate():
    assert abs(pass_at_k(100, 2, 1) - 0.02) < 1e-12


def test_pass_at_k_is_one_when_failures_cannot_fill_k():
    assert pass_at_k(8, 2, 8) == 1.0
    assert pass_at_k(8, 0, 1) == 0.0


def test_pass_at_k_from_flags():
    flags = [False, True, False, False]
    metrics = pass_at_k_from_flags(flags, [1, 4])
    assert abs(metrics[1] - 0.25) < 1e-12
    assert metrics[4] == 1.0
