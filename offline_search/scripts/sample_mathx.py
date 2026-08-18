#!/usr/bin/env python
"""Stream Modotte/MathX-5M (problem, expected_answer) into a 2000-row parquet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.utils.io import records_to_parquet

DATASET = "Modotte/MathX-5M"
DEFAULT_N = 2000
DEFAULT_SEED = 42
DEFAULT_PARQUET = ROOT / "data" / "mathx5m" / "mathx5m_2000.parquet"
DEFAULT_JSON = ROOT / "examples" / "qwen25_1p5b" / "fixtures" / "mathx5m_2000.json"
MAX_PROBLEM_CHARS = 6000


def _cell(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value).strip()


def stream_mathx_rows(
    *,
    n: int = DEFAULT_N,
    max_problem_chars: int = MAX_PROBLEM_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    rows: list[dict[str, Any]] = []
    skipped = {"empty": 0, "long": 0}
    for source_index, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        problem = _cell(row, "problem")
        answer = _cell(row, "expected_answer")
        if not problem or not answer:
            skipped["empty"] += 1
            continue
        if len(problem) > int(max_problem_chars):
            skipped["long"] += 1
            continue
        kept = len(rows)
        rows.append(
            {
                "problem_id": f"mathx5m/train/{kept:06d}",
                "problem": problem,
                "expected_answer": answer,
                "source_index": int(source_index),
            }
        )
        if kept + 1 >= int(n):
            break
        if (kept + 1) % 200 == 0:
            print(f"streamed {kept + 1}/{n} (skipped empty={skipped['empty']} long={skipped['long']})", flush=True)
    if len(rows) < int(n):
        raise SystemExit(f"Only streamed {len(rows)} usable rows; wanted {n}")
    return rows, skipped


def to_problem_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        answer = row["expected_answer"]
        records.append(
            {
                "problem_id": row["problem_id"],
                "problem_text": row["problem"],
                "problem": row["problem"],
                "ground_truth": answer,
                "answer": answer,
                "expected_answer": answer,
                "source_index": row.get("source_index"),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream MathX-5M into a 2000-row parquet + problems JSON.")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-problem-chars", type=int, default=MAX_PROBLEM_CHARS)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    print(f"streaming {args.n} rows from {DATASET} columns=problem,expected_answer", flush=True)
    rows, skipped = stream_mathx_rows(n=int(args.n), max_problem_chars=int(args.max_problem_chars))
    parquet_rows = [{"problem": r["problem"], "expected_answer": r["expected_answer"]} for r in rows]
    args.parquet.parent.mkdir(parents=True, exist_ok=True)
    records_to_parquet(args.parquet, parquet_rows)

    payload = {
        "meta": {
            "dataset": DATASET,
            "split": "train",
            "n": int(args.n),
            "seed": int(args.seed),
            "num_records": len(rows),
            "columns": ["problem", "expected_answer"],
            "max_problem_chars": int(args.max_problem_chars),
            "skipped_empty": skipped["empty"],
            "skipped_long": skipped["long"],
            "parquet": str(args.parquet),
        },
        "records": to_problem_records(rows),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {len(rows)} rows parquet={args.parquet} json={args.json_out} "
        f"skipped_empty={skipped['empty']} skipped_long={skipped['long']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
