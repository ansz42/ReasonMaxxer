from __future__ import annotations

from offline_search.data.select_trajectories import SelectionCaps, select_trajectories


def _row(pid: str, reward: float, correct: bool = False, near: bool = False, seed: int = 0) -> dict:
    return {
        "problem_id": pid,
        "reward": reward,
        "is_correct": correct,
        "near_correct": near,
        "sampling_config_id": "cfg",
        "seed": seed,
        "response": f"{pid}-{reward}-{seed}",
    }


def test_keeps_all_correct_up_to_cap_and_limits_garbage():
    rows = []
    rows += [_row("p", 1.0, correct=True, seed=i) for i in range(12)]
    rows += [_row("p", 0.85, near=True, seed=100 + i) for i in range(4)]
    rows += [_row("p", 0.4, seed=200 + i) for i in range(10)]
    rows += [_row("p", 0.0, seed=300 + i) for i in range(50)]
    selected = select_trajectories(
        rows,
        SelectionCaps(
            max_correct_per_problem=8,
            max_near_correct_per_problem=4,
            max_hard_negatives_per_problem=5,
            max_low_reward_negatives_per_problem=2,
        ),
    )
    assert len([r for r in selected if r["is_correct"]]) == 8
    assert len(selected) == 8 + 4 + 5 + 2
    leftover = [r for r in selected if not r["is_correct"]]
    # Top near-correct fill first; leftover high-reward incorrects become hard negatives.
    assert leftover[0]["reward"] == 0.85
    assert any(r["reward"] == 0.4 for r in leftover)
    assert sum(1 for r in leftover if r["reward"] == 0.0) == 2


def test_selection_is_per_problem():
    rows = [_row("a", 1.0, correct=True, seed=1), _row("b", 1.0, correct=True, seed=2), _row("b", 0.0, seed=3)]
    selected = select_trajectories(rows, SelectionCaps(1, 0, 1, 0))
    assert {r["problem_id"] for r in selected} == {"a", "b"}
