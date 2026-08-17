from __future__ import annotations

from offline_search.utils.tracking import (
    search_progress_metrics,
    should_log_search_progress,
    summarize_search_records,
)


def test_should_log_search_progress_on_first_and_every_eighth():
    logged = [i for i in range(0, 25) if should_log_search_progress(i)]
    assert logged == [1, 8, 16, 24]


def test_search_progress_metrics_use_rollout_step():
    payload = search_progress_metrics(
        16,
        {"reward": 0.4, "is_correct": False, "generated_tokens": 10},
        tokens=800,
        reward_sum=10.0,
        correct=8,
    )
    assert payload["search/step"] == 16
    assert payload["search/rollouts"] == 16
    assert payload["search/tokens"] == 800
    assert payload["search/reward"] == 0.4
    assert payload["search/reward_mean"] == 0.625
    assert payload["search/is_correct"] == 0
    assert payload["search/correct_rate"] == 0.5


def test_summarize_search_records_counts_existing_jsonl():
    stats = summarize_search_records(
        [
            {"reward": 1.0, "is_correct": True, "generated_tokens": 12},
            {"reward": 0.4, "is_correct": False, "generated_tokens": 8},
        ]
    )
    assert stats["n"] == 2
    assert stats["tokens"] == 20
    assert stats["correct"] == 1
    assert stats["reward_mean"] == 0.7
    assert stats["correct_rate"] == 0.5
