from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

from offline_search.search.clipping import is_clipped_record


@dataclass(frozen=True)
class SelectionCaps:
    max_correct_per_problem: int = 8
    max_near_correct_per_problem: int = 16
    max_hard_negatives_per_problem: int = 32
    max_low_reward_negatives_per_problem: int = 4
    drop_clipped: bool = True
    max_generated_tokens: int | None = None


def _reward(row: dict[str, Any]) -> float:
    return float(row.get("reward", 0.0))


def _is_correct(row: dict[str, Any]) -> bool:
    return bool(row.get("is_correct", False))


def _is_near(row: dict[str, Any]) -> bool:
    return bool(row.get("near_correct", False)) and not _is_correct(row)


def _stable_sort(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            -_reward(r),
            str(r.get("sampling_config_id", "")),
            int(r.get("seed", 0)),
            str(r.get("response", "")),
        ),
    )


def select_for_problem(rows: Sequence[dict[str, Any]], caps: SelectionCaps) -> list[dict[str, Any]]:
    correct = _stable_sort([r for r in rows if _is_correct(r)])
    near = _stable_sort([r for r in rows if _is_near(r)])
    incorrect = [r for r in rows if not _is_correct(r)]

    selected: list[dict[str, Any]] = []
    used: set[int] = set()

    def take(candidates: Sequence[dict[str, Any]], limit: int) -> None:
        taken = 0
        for row in candidates:
            ident = id(row)
            if ident in used:
                continue
            if taken >= limit:
                break
            selected.append(row)
            used.add(ident)
            taken += 1

    take(correct, int(caps.max_correct_per_problem))
    take(near, int(caps.max_near_correct_per_problem))

    # Hard negatives: incorrect, relatively high reward, not already taken as near-correct.
    hard_pool = _stable_sort([r for r in incorrect if id(r) not in used])
    take(hard_pool, int(caps.max_hard_negatives_per_problem))

    # A few lowest-reward leftovers for diversity.
    leftovers = sorted(
        [r for r in incorrect if id(r) not in used],
        key=lambda r: (
            _reward(r),
            str(r.get("sampling_config_id", "")),
            int(r.get("seed", 0)),
        ),
    )
    take(leftovers, int(caps.max_low_reward_negatives_per_problem))
    return selected


def select_trajectories(rows: Sequence[dict[str, Any]], caps: SelectionCaps | None = None) -> list[dict[str, Any]]:
    caps = caps or SelectionCaps()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("problem_id", ""))].append(row)

    selected: list[dict[str, Any]] = []
    for problem_id in sorted(grouped):
        rows = grouped[problem_id]
        if caps.drop_clipped:
            rows = [r for r in rows if not is_clipped_record(r, caps.max_generated_tokens)]
        else:
            rows = [r for r in rows if not r.get("discarded")]
        if rows:
            selected.extend(select_for_problem(rows, caps))
    return selected
