from __future__ import annotations

import ast
import math
import re
from typing import Any

from offline_search.scoring.base import ScoreResult

BOXED_RE = re.compile(r"\\boxed\s*\{")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z_])")


def _extract_braced_group(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def extract_boxed_answer(text: str) -> str | None:
    if not text:
        return None
    matches = list(BOXED_RE.finditer(text))
    for match in reversed(matches):
        group = _extract_braced_group(text, match.end() - 1)
        if group is not None:
            return group.strip()
    return None


def extract_final_number(text: str) -> str | None:
    if not text:
        return None
    boxed = extract_boxed_answer(text)
    sources = [boxed, text] if boxed else [text]
    for src in sources:
        if not src:
            continue
        tokens = NUMBER_RE.findall(src)
        if tokens:
            return tokens[-1].replace(",", "")
    return None


def extract_math_answer(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    return extract_final_number(text)


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    text = answer.strip()
    if not text:
        return None
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\:", "")
    text = text.replace("\\times", "*").replace("\\cdot", "*")
    text = text.replace("\\pi", "pi").replace("π", "pi")
    # Convert \frac before stripping braces, otherwise \frac{7}{8} becomes \frac78 -> 78.
    text = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\frac\s*([^\s/{]+)\s*([^\s/{]+)", r"(\1)/(\2)", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\([a-zA-Z]+)", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    if "=" in text:
        text = text.split("=")[-1]
    return text.strip(" .,:;") or None


def _safe_float(expr: str | None) -> float | None:
    if not expr:
        return None
    cleaned = expr.replace(",", "").replace(" ", "")
    try:
        value = float(cleaned)
    except ValueError:
        try:
            parsed = ast.literal_eval(cleaned)
            value = float(parsed)
        except Exception:
            if "/" in cleaned:
                left, _, right = cleaned.partition("/")
                left_v = _safe_float(left.strip("()"))
                right_v = _safe_float(right.strip("()"))
                if left_v is None or right_v is None or right_v == 0.0:
                    return None
                value = left_v / right_v
            else:
                return None
    if not math.isfinite(value):
        return None
    return value


def numeric_relative_error(predicted: str | None, reference: str | None) -> float | None:
    pred = _safe_float(normalize_answer(predicted))
    gold = _safe_float(normalize_answer(reference))
    if pred is None or gold is None:
        return None
    denom = max(abs(gold), 1e-8)
    return abs(pred - gold) / denom


def answers_match(predicted: str | None, reference: str | None, *, tol: float = 1e-6) -> bool:
    pred = normalize_answer(predicted)
    gold = normalize_answer(reference)
    if pred is None or gold is None:
        return False
    if pred == gold:
        return True
    pred_v = _safe_float(pred)
    gold_v = _safe_float(gold)
    if pred_v is None or gold_v is None:
        return False
    return abs(pred_v - gold_v) <= tol


class MathVerifier:
    """Scalar graded math scorer. Categories stay in the scorer, not the trainer."""

    def __init__(
        self,
        *,
        near_rel_error: float = 0.05,
        approach_rel_error: float = 0.20,
    ) -> None:
        self.near_rel_error = float(near_rel_error)
        self.approach_rel_error = float(approach_rel_error)

    def score_rollout(self, prompt: str, response: str, reference: str | None = None) -> ScoreResult:
        del prompt
        text = response or ""
        extracted = extract_math_answer(text)
        metadata: dict[str, Any] = {"predicted": extracted, "reference": reference}

        if not text.strip():
            return ScoreResult(0.0, False, False, metadata)

        if extracted is None:
            return ScoreResult(0.15, False, False, metadata)

        if answers_match(extracted, reference):
            return ScoreResult(1.0, True, False, metadata)

        rel = numeric_relative_error(extracted, reference)
        metadata["relative_error"] = rel
        if rel is not None and rel <= self.near_rel_error:
            return ScoreResult(0.85, False, True, metadata)
        if rel is not None and rel <= self.approach_rel_error:
            return ScoreResult(0.65, False, True, metadata)
        return ScoreResult(0.40, False, False, metadata)
