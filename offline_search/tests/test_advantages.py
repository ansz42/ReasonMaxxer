from __future__ import annotations

from offline_search.data.advantages import attach_advantages, per_problem_advantages


def test_high_reward_is_positive_and_low_is_negative():
    adv = per_problem_advantages([1.0, 0.4, 0.0])
    assert adv[0] > 0
    assert adv[2] < 0
    assert abs(sum(adv)) < 1e-9


def test_constant_rewards_have_zero_advantage():
    adv = per_problem_advantages([0.5, 0.5, 0.5])
    assert all(abs(x) < 1e-9 for x in adv)


def test_advantages_are_normalized_per_problem_not_globally():
    rows = [
        {"problem_id": "a", "reward": 1.0},
        {"problem_id": "a", "reward": 0.0},
        {"problem_id": "b", "reward": 0.2},
        {"problem_id": "b", "reward": 0.1},
    ]
    out = attach_advantages(rows)
    a = [r for r in out if r["problem_id"] == "a"]
    b = [r for r in out if r["problem_id"] == "b"]
    assert a[0]["advantage"] > 0 > a[1]["advantage"]
    assert b[0]["advantage"] > 0 > b[1]["advantage"]
    # The 0.2 reward on problem b is the local best, even though globally small.
    assert b[0]["advantage"] > 0
