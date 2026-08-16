from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SearchAccounting:
    generated_trajectories: int = 0
    generated_tokens: int = 0
    search_wall_time_s: float = 0.0
    training_tokens: int = 0
    training_steps: int = 0
    training_wall_time_s: float = 0.0
    gpu_hours: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    _t0: float | None = field(default=None, repr=False, compare=False)

    def start_timer(self) -> None:
        self._t0 = time.perf_counter()

    def stop_search_timer(self) -> None:
        if self._t0 is None:
            return
        self.search_wall_time_s += time.perf_counter() - self._t0
        self._t0 = None

    def stop_train_timer(self) -> None:
        if self._t0 is None:
            return
        self.training_wall_time_s += time.perf_counter() - self._t0
        self._t0 = None

    def add_rollout(self, tokens: int) -> None:
        self.generated_trajectories += 1
        self.generated_tokens += int(tokens)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("_t0", None)
        extra = payload.pop("extra", {})
        payload.update(extra)
        if self.search_wall_time_s > 0:
            payload["generated_tokens_per_s"] = self.generated_tokens / self.search_wall_time_s
        return payload

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
