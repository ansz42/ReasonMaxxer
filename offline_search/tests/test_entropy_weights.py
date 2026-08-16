from __future__ import annotations

from offline_search.data.entropy_weights import entropy_weights


def test_sigmoid_is_half_at_threshold():
    weights = entropy_weights([0.8], threshold=0.8, scale=0.25, mode="sigmoid")
    assert abs(weights[0] - 0.5) < 1e-9


def test_high_entropy_gets_stronger_weight():
    low, mid, high = entropy_weights([0.1, 0.8, 2.0], threshold=0.8, scale=0.25, mode="sigmoid")
    assert low < mid < high
    assert high > 0.95
    assert low < 0.06


def test_hard_mask():
    weights = entropy_weights([0.2, 0.8, 1.1], threshold=0.8, scale=0.25, mode="hard")
    assert weights == [0.0, 0.0, 1.0]
