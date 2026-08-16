from __future__ import annotations

from offline_search.scoring.base import ScoreResult


class ExactMatchScorer:
    def __init__(self, *, strip: bool = True, lowercase: bool = False) -> None:
        self.strip = strip
        self.lowercase = lowercase

    def _norm(self, text: str | None) -> str | None:
        if text is None:
            return None
        value = text.strip() if self.strip else text
        if self.lowercase:
            value = value.lower()
        return value

    def score_rollout(self, prompt: str, response: str, reference: str | None = None) -> ScoreResult:
        del prompt
        pred = self._norm(response)
        gold = self._norm(reference)
        correct = pred is not None and gold is not None and pred == gold and pred != ""
        return ScoreResult(
            reward=1.0 if correct else 0.0,
            is_correct=correct,
            near_correct=False,
            metadata={"predicted": pred, "reference": gold},
        )
