from __future__ import annotations

from offline_search.data.build_training_dataset import CharTokenizer, build_training_rows
from offline_search.data.select_trajectories import SelectionCaps


def _rec(pid: str, response: str, reward: float, correct: bool, near: bool = False, seed: int = 0) -> dict:
    return {
        "problem_id": pid,
        "prompt": "Q?",
        "response": response,
        "reward": reward,
        "is_correct": correct,
        "near_correct": near,
        "sampling_config_id": "cfg",
        "seed": seed,
    }


def test_successful_sft_keeps_only_correct_and_positive_advantage():
    rows = [
        _rec("p", "good", 1.0, True, seed=1),
        _rec("p", "bad", 0.0, False, seed=2),
    ]
    built = build_training_rows(rows, tokenizer=CharTokenizer(), objective="successful_sft")
    assert len(built) == 1
    assert built[0]["advantage"] == 1.0
    assert all(w == m for w, m in zip(built[0]["token_weight"], built[0]["response_mask"]))


def test_positive_only_clamps_negative_advantages():
    rows = [
        _rec("p", "good", 1.0, True, seed=1),
        _rec("p", "ok", 0.4, False, seed=2),
        _rec("p", "bad", 0.0, False, seed=3),
    ]
    built = build_training_rows(
        rows,
        tokenizer=CharTokenizer(),
        objective="positive_only",
        caps=SelectionCaps(8, 8, 8, 8),
    )
    assert all(r["advantage"] >= 0.0 for r in built)
    assert any(r["advantage"] > 0.0 for r in built)


def test_prompt_tokens_are_masked_and_labels_ignore_prompt():
    rows = [_rec("p", "ANS", 1.0, True)]
    built = build_training_rows(rows, tokenizer=CharTokenizer(), objective="graded_signed")
    row = built[0]
    assert row["response_mask"][: row["prompt_length"]] == [0] * row["prompt_length"]
    assert all(x == -100 for x in row["labels"][: row["prompt_length"]])
    assert sum(row["response_mask"]) > 0
