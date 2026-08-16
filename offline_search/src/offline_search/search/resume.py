from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence


JobKey = tuple[str, str, int]


def job_key(problem_id: str, sampling_config_id: str, seed: int) -> JobKey:
    return (str(problem_id), str(sampling_config_id), int(seed))


def record_job_key(record: dict[str, Any]) -> JobKey:
    return job_key(
        str(record["problem_id"]),
        str(record.get("sampling_config_id") or record.get("config_id")),
        int(record["seed"]),
    )


def completed_job_keys(records: Iterable[dict[str, Any]]) -> set[JobKey]:
    keys: set[JobKey] = set()
    for record in records:
        try:
            keys.add(record_job_key(record))
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def filter_pending(jobs: Sequence[Any], completed: set[JobKey]) -> list[Any]:
    pending: list[Any] = []
    for job in jobs:
        if hasattr(job, "key"):
            key = job.key
        else:
            key = job_key(job["problem_id"], job["sampling_config_id"], int(job["seed"]))
        if key not in completed:
            pending.append(job)
    return pending


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def append_jsonl(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
