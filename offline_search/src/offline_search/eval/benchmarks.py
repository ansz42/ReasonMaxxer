from __future__ import annotations

from typing import Any, Callable, Sequence

from offline_search.data.aime24 import AIME24_HF_NAME, load_aime24_records
from offline_search.data.gsm8k import GSM8K_HF_NAME, load_gsm8k_records
from offline_search.data.math500 import MATH500_HF_NAME, load_math500_rows, row_to_record as math500_row_to_record
from offline_search.prompting import build_user_prompt
from offline_search.search.search_runner import Problem


def problems_from_records(
    records: Sequence[dict[str, Any]],
    *,
    prompt_style: str = "qwen3_chat",
) -> list[Problem]:
    problems: list[Problem] = []
    for i, row in enumerate(records):
        text = row.get("problem_text") or row.get("problem") or row.get("question") or ""
        gold = row.get("ground_truth")
        if gold is None:
            gold = row.get("reference_answer")
        if gold is None:
            gold = row.get("answer")
        pid = str(row.get("problem_id") or f"problem/{i}")
        problems.append(
            Problem(
                problem_id=pid,
                prompt=build_user_prompt(str(text), prompt_style=prompt_style),
                reference_answer=None if gold is None else str(gold),
                extra=dict(row),
            )
        )
    return problems


def load_math500_records(*, limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_math500_rows()
    records = [math500_row_to_record(row, i) for i, row in enumerate(rows)]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return records


BENCHMARK_LOADERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "gsm8k": load_gsm8k_records,
    "math500": load_math500_records,
    "aime24": load_aime24_records,
}

BENCHMARK_SOURCES = {
    "gsm8k": GSM8K_HF_NAME,
    "math500": MATH500_HF_NAME,
    "aime24": AIME24_HF_NAME,
}


def load_benchmark_problems(
    name: str,
    *,
    prompt_style: str = "qwen3_chat",
    limit: int | None = None,
) -> list[Problem]:
    key = name.strip().lower()
    if key not in BENCHMARK_LOADERS:
        known = ", ".join(sorted(BENCHMARK_LOADERS))
        raise ValueError(f"Unknown benchmark {name!r}. Known: {known}")
    records = BENCHMARK_LOADERS[key](limit=limit)
    return problems_from_records(records, prompt_style=prompt_style)
