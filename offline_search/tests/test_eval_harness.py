from __future__ import annotations

from offline_search.data.aime24 import aime24_row_to_record
from offline_search.data.gsm8k import extract_gsm8k_gold, gsm8k_row_to_record
from offline_search.data.math500 import row_to_record as math500_row_to_record
from offline_search.eval.benchmarks import BENCHMARK_LOADERS, problems_from_records
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
    assert report["avg@1"] == 1.0
    assert report["num_problems"] == 2


def test_aime24_row_to_record_keeps_official_answer():
    rec = aime24_row_to_record({"id": 67, "problem": "Find xy.", "answer": "025"}, 0)
    assert rec["problem_id"] == "aime24/67"
    assert rec["problem_text"] == "Find xy."
    assert rec["ground_truth"] == "025"
    assert rec["answer"] == "025"
    assert MathVerifier().score_rollout("q", r"\boxed{25}", rec["ground_truth"]).is_correct
    assert "aime24" in BENCHMARK_LOADERS


class _CannedBackend:
    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.i = 0

    def generate(self, prompts, **kwargs):
        del kwargs
        out = []
        for _ in prompts:
            out.append([GenerationResult(text=self.texts[self.i], num_tokens=3)])
            self.i += 1
        return out


def test_harness_avg_at_8():
    # 3/8 correct -> avg@8 = 0.375, pass@8 = 1.
    texts = [r"\boxed{4}"] * 3 + [r"\boxed{0}"] * 5
    report = evaluate_backend(
        [Problem("a", "q", "4")],
        _CannedBackend(texts),
        MathVerifier(),
        n_samples=8,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16,
        ks=[1, 8],
        generation_batch_size=8,
    )
    assert report["n_samples"] == 8
    assert report["avg@8"] == 0.375
    assert report["micro_correct_rate"] == 0.375
    assert report["macro"]["pass@1"] == 0.375
    assert report["macro"]["pass@8"] == 1.0
    assert report["per_problem"][0]["n_correct"] == 3
