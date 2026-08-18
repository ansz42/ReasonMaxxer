from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

AIME24_HF_NAME = "HuggingFaceH4/aime_2024"
AIME24_SPLIT = "train"


def _repo_benchmark_path() -> Path:
    # offline_search/src/offline_search/data/aime24.py -> ReasonMaxxer-fork/data/benchmarks/aime24.json
    return Path(__file__).resolve().parents[4] / "data" / "benchmarks" / "aime24.json"


def aime24_row_to_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    raw_id = row.get("problem_id") or row.get("id") or row.get("ID") or index
    problem = row.get("problem_text") or row.get("problem") or row.get("Problem") or row.get("question") or ""
    answer = row.get("ground_truth")
    if answer is None:
        answer = row.get("answer")
    if answer is None:
        answer = row.get("Answer")
    gold = "" if answer is None else str(answer).strip()
    return {
        "problem_id": f"aime24/{raw_id}",
        "problem_text": str(problem),
        "ground_truth": gold,
        "answer": gold,
        "solution": str(row.get("solution") or row.get("Solution") or ""),
        "year": str(row.get("year") or "2024"),
        "url": row.get("url"),
        "source_index": int(index),
    }


def _records_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("records") or payload.get("data") or []
    else:
        rows = payload
    if not isinstance(rows, list):
        raise ValueError(f"AIME24 JSON must be a list or contain records: {path}")
    return [aime24_row_to_record(dict(row), i) for i, row in enumerate(rows) if isinstance(row, dict)]


def load_aime24_rows(*, limit: int | None = None) -> list[dict[str, Any]]:
    env_path = os.environ.get("REASONMAXXER_AIME24")
    candidates = [Path(env_path)] if env_path else []
    candidates.append(_repo_benchmark_path())
    for path in candidates:
        if path.is_file():
            records = _records_from_json(path)
            break
    else:
        from datasets import load_dataset

        split = load_dataset(AIME24_HF_NAME, split=AIME24_SPLIT)
        records = [aime24_row_to_record(dict(row), i) for i, row in enumerate(split)]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return records


def load_aime24_records(
    rows: Sequence[dict[str, Any]] | None = None,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if rows is None:
        return load_aime24_rows(limit=limit)
    records = [aime24_row_to_record(dict(row), i) for i, row in enumerate(rows)]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return records
