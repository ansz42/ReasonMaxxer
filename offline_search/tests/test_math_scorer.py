from __future__ import annotations

from offline_search.scoring.exact_match import ExactMatchScorer
from offline_search.scoring.math_verifier import (
    MathVerifier,
    answers_match,
    extract_boxed_answer,
    extract_math_answer,
)


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


def test_extracts_from_last_two_lines_not_earlier_numbers():
    text = (
        "I first compute 17 + 28 = 45.\n"
        "That 45 is only an intermediate check.\n"
        "The answer is 12\n"
        "I mentioned 45 earlier as a sanity check."
    )
    assert extract_math_answer(text) == "12"
    scorer = MathVerifier()
    result = scorer.score_rollout("q", text, "12")
    assert result.is_correct and result.reward == 1.0


def test_extracts_unboxed_frac_from_final_answer_line():
    text = "I cancelled a 99 along the way.\nFinal answer: $\\frac{1}{2}$"
    extracted = extract_math_answer(text)
    assert answers_match(extracted, "1/2")
    scorer = MathVerifier()
    assert scorer.score_rollout("q", text, r"\frac{1}{2}").is_correct


def test_extracts_bare_last_line_expression():
    text = "After simplifying we obtain\n7/8"
    extracted = extract_math_answer(text)
    assert answers_match(extracted, "7/8")


def test_prefers_last_two_line_boxed_over_earlier_boxed():
    text = "A first guess is \\boxed{9}.\nLater correction.\nSo \\boxed{4}"
    assert extract_math_answer(text) == "4"


def test_ignores_trailing_asy_diagram_after_boxed():
    text = (
        "The perimeter is $2(21)=\\boxed{42}$ inches.\n\n"
        "[asy]\n"
        "draw(dir(60*i)--dir(60*(i+3)));\n"
        "[/asy]"
    )
    assert extract_math_answer(text) == "42"
