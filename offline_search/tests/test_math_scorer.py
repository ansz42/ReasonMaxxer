from __future__ import annotations

from offline_search.scoring.exact_match import ExactMatchScorer
from offline_search.scoring.math_verifier import MathVerifier, extract_boxed_answer


def test_boxed_extraction():
    assert extract_boxed_answer("work \\boxed{42}") == "42"


def test_exact_match_scorer():
    scorer = ExactMatchScorer()
    assert scorer.score_rollout("q", "42", "42").is_correct
    assert not scorer.score_rollout("q", "41", "42").is_correct


def test_math_verifier_grades_answers():
    scorer = MathVerifier()
    correct = scorer.score_rollout("q", "Therefore \\boxed{45}", "45")
    near = scorer.score_rollout("q", "\\boxed{46}", "45")
    wrong = scorer.score_rollout("q", "I think \\boxed{1}", "45")
    empty = scorer.score_rollout("q", "   ", "45")
    prose = scorer.score_rollout("q", "This is a hard problem.", "45")
    assert correct.reward == 1.0 and correct.is_correct
    assert near.near_correct and 0.6 < near.reward < 1.0
    assert wrong.reward == 0.40 and not wrong.is_correct
    assert empty.reward == 0.0
    assert prose.reward == 0.15


def test_math_verifier_accepts_latex_frac():
    scorer = MathVerifier()
    boxed = scorer.score_rollout("q", r"Final answer: $$\boxed{\frac{7}{8}}$$", "7/8")
    assert boxed.is_correct and boxed.reward == 1.0
    slash = scorer.score_rollout("q", r"\boxed{7/8}", r"\frac{7}{8}")
    assert slash.is_correct
