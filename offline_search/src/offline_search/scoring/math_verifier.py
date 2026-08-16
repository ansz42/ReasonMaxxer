from __future__ import annotations

import ast
import math
import re
from typing import Any

from offline_search.scoring.base import ScoreResult

BOXED_RE = re.compile(r"\\boxed\s*\{")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?(?![A-Za-z_])")
# Only the last couple of lines are treated as the answer region. Intermediate
# arithmetic in the writeup is ignored so we can parse that tail more freely.
TAIL_LINE_COUNT = 2
ANSWER_CUE_RE = re.compile(
    r"(?:"
    r"final\s+answer(?:\s+is)?"
    r"|the\s+answer\s+is"
    r"|answer\s*(?:is|:)"
    r"|thus,?\s+(?:the\s+)?answer(?:\s+is)?"
    r"|therefore,?\s+(?:the\s+)?answer(?:\s+is)?"
    r"|so,?\s+the\s+answer(?:\s+is)?"
    r")\s*[:\-]?\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
DOLLAR_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.DOTALL)
LATEX_FRAC_RE = re.compile(r"\\(?:d|t|c)?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
SLASH_FRAC_RE = re.compile(r"(?<![A-Za-z0-9.])-?\d+\s*/\s*-?\d+(?![A-Za-z0-9.])")
PROSE_WORD_RE = re.compile(r"(?<!\\)[A-Za-z]{3,}")
ASY_BLOCK_RE = re.compile(r"\[asy\].*?\[/asy\]", re.IGNORECASE | re.DOTALL)


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


def last_nonempty_lines(text: str, n: int = TAIL_LINE_COUNT) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines[-max(1, int(n)) :])


def strip_diagrams(text: str) -> str:
    return ASY_BLOCK_RE.sub("", text or "")


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


def _latex_frac_to_slash(text: str) -> str | None:
    match = LATEX_FRAC_RE.search(text)
    if not match:
        return None
    return f"{match.group(1).strip()}/{match.group(2).strip()}"


def _cleanup_candidate(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = raw.strip().strip(" .,:;")
    if not text:
        return None
    if text.startswith("$$") and text.endswith("$$") and len(text) > 4:
        text = text[2:-2].strip()
    text = text.strip("$").strip()
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed.strip()
    frac = _latex_frac_to_slash(text)
    if frac:
        return frac
    slash = SLASH_FRAC_RE.search(text)
    if slash:
        return slash.group(0).replace(" ", "")
    if PROSE_WORD_RE.search(text):
        first = NUMBER_RE.search(text)
        if first:
            return first.group(0).replace(",", "")
        return None
    return text or None


def extract_from_region(region: str) -> str | None:
    if not region or not region.strip():
        return None

    boxed = extract_boxed_answer(region)
    if boxed:
        return boxed

    cues = list(ANSWER_CUE_RE.finditer(region))
    if cues:
        candidate = _cleanup_candidate(cues[-1].group(1))
        if candidate:
            return candidate

    dollars: list[str] = []
    for match in DOLLAR_MATH_RE.finditer(region):
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        if inner and "\n" not in inner:
            dollars.append(inner)
    if dollars:
        candidate = _cleanup_candidate(dollars[-1])
        if candidate:
            return candidate

    frac = None
    for match in LATEX_FRAC_RE.finditer(region):
        frac = f"{match.group(1).strip()}/{match.group(2).strip()}"
    if frac:
        return frac

    slashes = SLASH_FRAC_RE.findall(region)
    if slashes:
        return slashes[-1].replace(" ", "")

    return extract_final_number(region)


def extract_math_answer(text: str) -> str | None:
    if not text:
        return None
    cleaned = strip_diagrams(text)
    boxed = extract_boxed_answer(cleaned) or extract_boxed_answer(text)
    if boxed:
        return boxed
    return extract_from_region(last_nonempty_lines(cleaned, TAIL_LINE_COUNT))


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
