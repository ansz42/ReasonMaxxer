from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

MATH500_HF_NAME = "HuggingFaceH4/MATH-500"
MATH500_SPLIT = "test"
MATH500_JSONL_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
)
DEFAULT_SAMPLE_SIZE = 300
DEFAULT_SEED = 42


def row_to_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    unique_id = str(row.get("unique_id") or f"idx{index}")
    subject = str(row.get("subject") or "unknown")
    answer = row.get("answer")
    return {
        "problem_id": f"math500/{MATH500_SPLIT}/{unique_id}",
        "problem_text": str(row.get("problem") or ""),
        "ground_truth": "" if answer is None else str(answer),
        "answer": "" if answer is None else str(answer),
        "solution": str(row.get("solution") or ""),
        "subject": subject,
        "level": row.get("level"),
        "unique_id": unique_id,
        "category": subject,
        "source_index": int(index),
    }


def sample_records(
    rows: Sequence[dict[str, Any]],
    *,
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if n > len(rows):
        raise ValueError(f"Requested {n} rows but only {len(rows)} are available")
    rng = random.Random(int(seed))
    chosen = rng.sample(list(range(len(rows))), k=int(n))
    chosen.sort()
    return [row_to_record(rows[i], i) for i in chosen]


def records_payload(
    records: Sequence[dict[str, Any]],
    *,
    n: int,
    seed: int,
    dataset: str = MATH500_HF_NAME,
    split: str = MATH500_SPLIT,
) -> dict[str, Any]:
    return {
        "meta": {
            "dataset": dataset,
            "split": split,
            "n": int(n),
            "seed": int(seed),
            "num_records": len(records),
        },
        "records": list(records),
    }


def write_records(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_math500_rows() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset

        split = load_dataset(MATH500_HF_NAME, split=MATH500_SPLIT)
        return [dict(row) for row in split]
    except Exception:
        return _download_math500_jsonl()


def _download_math500_jsonl(url: str = MATH500_JSONL_URL) -> list[dict[str, Any]]:
    with urllib.request.urlopen(url, timeout=120) as response:
        raw = response.read().decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    if not rows:
        raise ValueError(f"No MATH-500 rows downloaded from {url}")
    return rows


def sample_math500(
    *,
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    material = list(rows) if rows is not None else load_math500_rows()
    records = sample_records(material, n=n, seed=seed)
    return records_payload(records, n=n, seed=seed)
