from __future__ import annotations

from offline_search.search.adaptive_allocator import allocate_remaining, config_score, softmax_allocation_probs


def test_config_score_weights_correctness_highest():
    high_correct = config_score(0.50, 0.0, 0.0)
    high_near = config_score(0.0, 1.0, 0.0)
    high_reward = config_score(0.0, 0.0, 1.0)
    assert high_correct > high_near > high_reward


def test_allocate_remaining_sums_to_budget():
    alloc = allocate_remaining([0.1, 2.0, 0.3, 0.2], 100, exploration_fraction=0.20)
    assert sum(alloc) == 100
    assert len(alloc) == 4


def test_better_configs_receive_more_budget():
    alloc = allocate_remaining([0.1, 3.0, 0.1, 0.1], 100, exploration_fraction=0.20, allocation_temperature=0.5)
    assert alloc[1] == max(alloc)
    assert alloc[1] > alloc[0]
    assert min(alloc) >= 1


def test_exploration_keeps_weak_configs_alive():
    alloc = allocate_remaining([10.0, 0.0, 0.0, 0.0], 100, exploration_fraction=0.20, allocation_temperature=0.1)
    assert alloc[0] >= 70
    assert min(alloc[1:]) >= 4


def test_zero_budget_is_all_zeros():
    assert allocate_remaining([1.0, 2.0], 0) == [0, 0]


def test_softmax_is_normalized():
    probs = softmax_allocation_probs([1.0, 2.0, 3.0], 0.5)
    assert abs(float(probs.sum()) - 1.0) < 1e-9
