from __future__ import annotations

import json
from pathlib import Path

from offline_search.data.math500 import sample_math500, sample_records
from offline_search.data.problems import load_problems_file
from offline_search.scoring.math_verifier import answers_match, extract_math_answer

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "qwen25_3b" / "fixtures" / "math500_300.json"
FIXTURE_500 = ROOT / "examples" / "qwen25_1p5b" / "fixtures" / "math500_500.json"


def _fake_rows(n: int = 10) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "problem": f"Compute {i}+1.",
                "solution": f"Adding one gives {i + 1}.\n\\boxed{{{i + 1}}}",
                "answer": str(i + 1),
                "subject": "Algebra" if i % 2 == 0 else "Geometry",
                "level": (i % 5) + 1,
                "unique_id": f"test/{i}.json",
            }
        )
    return rows


def test_sample_records_is_deterministic_and_sized():
    rows = _fake_rows(20)
    a = sample_records(rows, n=8, seed=42)
    b = sample_records(rows, n=8, seed=42)
    c = sample_records(rows, n=8, seed=7)
    assert len(a) == 8
    assert [r["problem_id"] for r in a] == [r["problem_id"] for r in b]
    assert [r["problem_id"] for r in a] != [r["problem_id"] for r in c]
    assert all(r["ground_truth"] == r["answer"] for r in a)
    assert all(r["problem_text"] and r["solution"] and r["subject"] for r in a)


def test_sample_math500_payload_shape():
    payload = sample_math500(n=5, seed=42, rows=_fake_rows(12))
    assert payload["meta"]["dataset"] == "HuggingFaceH4/MATH-500"
    assert payload["meta"]["split"] == "test"
    assert payload["meta"]["n"] == 5
    assert payload["meta"]["seed"] == 42
    assert len(payload["records"]) == 5


def test_math500_300_fixture_exists_and_loads():
    assert FIXTURE.exists(), f"Missing fixture {FIXTURE}; run scripts/sample_math500.py"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    records = payload["records"]
    assert payload["meta"]["dataset"] == "HuggingFaceH4/MATH-500"
    assert payload["meta"]["split"] == "test"
    assert payload["meta"]["n"] == 300
    assert payload["meta"]["seed"] == 42
    assert len(records) == 300
    ids = [row["problem_id"] for row in records]
    assert len(set(ids)) == 300
    for row in records:
        assert row["problem_text"]
        assert row.get("ground_truth") or row.get("answer")
        assert row.get("subject")
        assert row.get("solution")
    problems = load_problems_file(FIXTURE)
    assert len(problems) == 300
    assert all(p.reference_answer for p in problems)


def test_math500_500_fixture_is_full_split():
    assert FIXTURE_500.exists(), f"Missing fixture {FIXTURE_500}; run sample_math500.py --n 500"
    payload = json.loads(FIXTURE_500.read_text(encoding="utf-8"))
    records = payload["records"]
    assert payload["meta"]["dataset"] == "HuggingFaceH4/MATH-500"
    assert payload["meta"]["split"] == "test"
    assert payload["meta"]["n"] == 500
    assert len(records) == 500
    ids = [row["problem_id"] for row in records]
    assert len(set(ids)) == 500
    problems = load_problems_file(FIXTURE_500)
    assert len(problems) == 500
    assert all(p.reference_answer for p in problems)


def test_regex_recovers_most_math500_gold_answers_from_solutions():
    assert FIXTURE.exists(), f"Missing fixture {FIXTURE}; run scripts/sample_math500.py"
    records = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]
    hits = 0
    misses: list[tuple[str, str | None, str]] = []
    for row in records:
        extracted = extract_math_answer(row["solution"])
        gold = str(row.get("answer") or row.get("ground_truth") or "")
        if answers_match(extracted, gold):
            hits += 1
        else:
            misses.append((row["problem_id"], extracted, gold))
    recall = hits / len(records)
    assert recall >= 0.95, (
        f"gold-answer recall {recall:.3f} ({hits}/{len(records)}); "
        f"first misses={misses[:8]}"
    )
