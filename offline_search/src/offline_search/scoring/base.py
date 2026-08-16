from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class ScoreResult:
    reward: float
    is_correct: bool
    near_correct: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def score_to_dict(result: ScoreResult) -> dict[str, Any]:
    return asdict(result)


class RolloutScorer(Protocol):
    def score_rollout(
        self,
        prompt: str,
        response: str,
        reference: str | None = None,
    ) -> ScoreResult:
        ...
