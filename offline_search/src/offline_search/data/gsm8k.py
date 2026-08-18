from __future__ import annotations

from typing import Any, Sequence

GSM8K_HF_NAME = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"


def extract_gsm8k_gold(answer: str) -> str:
    text = str(answer or "").strip()
    if "####" in text:
        text = text.rsplit("####", 1)[-1].strip()
    return text.replace(",", "")


def gsm8k_row_to_record(row: dict[str, Any], index: int, *, split: str = GSM8K_SPLIT) -> dict[str, Any]:
    gold = extract_gsm8k_gold(str(row.get("answer") or row.get("ground_truth") or ""))
    question = str(row.get("question") or row.get("problem") or row.get("problem_text") or "")
    return {
        "problem_id": f"gsm8k/{split}/{index}",
        "problem_text": question,
        "ground_truth": gold,
        "answer": gold,
        "solution": str(row.get("answer") or ""),
        "source_index": int(index),
    }


def load_gsm8k_rows(*, split: str = GSM8K_SPLIT, limit: int | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(GSM8K_HF_NAME, GSM8K_CONFIG, split=split)
    rows = [gsm8k_row_to_record(dict(row), i, split=split) for i, row in enumerate(dataset)]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def load_gsm8k_records(
    rows: Sequence[dict[str, Any]] | None = None,
    *,
    split: str = GSM8K_SPLIT,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if rows is None:
        return load_gsm8k_rows(split=split, limit=limit)
    records = [gsm8k_row_to_record(dict(row), i, split=split) for i, row in enumerate(rows)]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return records
