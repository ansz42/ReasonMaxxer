from __future__ import annotations

from offline_search.data.gsm8k import extract_gsm8k_gold, gsm8k_row_to_record
from offline_search.data.math500 import row_to_record as math500_row_to_record
from offline_search.eval.benchmarks import problems_from_records
from offline_search.search.generate import GenerationResult
from offline_search.search.search_runner import Problem
from offline_search.eval.generate_eval import evaluate_backend
from offline_search.scoring.math_verifier import MathVerifier


def test_extract_gsm8k_gold_uses_hash_marker():
    raw = "She sold 48 in April and 24 in May.\n#### 72"
    assert extract_gsm8k_gold(raw) == "72"
    assert extract_gsm8k_gold("#### 1,234") == "1234"
    assert extract_gsm8k_gold("42") == "42"


def test_gsm8k_row_to_record():
    rec = gsm8k_row_to_record({"question": "What is 2+2?", "answer": "Adding.\n#### 4"}, 7)
    assert rec["problem_id"] == "gsm8k/test/7"
    assert rec["problem_text"] == "What is 2+2?"
    assert rec["ground_truth"] == "4"
    assert rec["answer"] == "4"


def test_problems_from_records_apply_chat_prompt():
    recs = [math500_row_to_record({"problem": "1+1", "answer": "2", "unique_id": "u"}, 0)]
    problems = problems_from_records(recs, prompt_style="qwen3_chat")
    assert len(problems) == 1
    assert problems[0].reference_answer == "2"
    assert "\\boxed{}" in problems[0].prompt
    assert "1+1" in problems[0].prompt


class _GreedyBackend:
    def generate(self, prompts, **kwargs):
        del kwargs
        return [[GenerationResult(text="\\boxed{4}", num_tokens=3)] for _ in prompts]


def test_harness_greedy_pass_at_1():
    problems = [
        Problem("a", "q", "4"),
        Problem("b", "q", "4"),
    ]
    report = evaluate_backend(
        problems,
        _GreedyBackend(),
        MathVerifier(),
        n_samples=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
        ks=[1],
        generation_batch_size=8,
    )
    assert report["macro"]["pass@1"] == 1.0
    assert report["micro_correct_rate"] == 1.0
    assert report["num_problems"] == 2
