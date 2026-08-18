from __future__ import annotations

from typing import Any

CLIP_FINISH_REASONS = frozenset({"length", "max_tokens", "length_capped"})


def clip_retry_seed(seed: int) -> int:
    """Deterministic alternate seed for one clipped-sample retry."""
    return int(seed) ^ 0x9E3779B9


def _finish_reason_clipped(reason: Any) -> bool:
    return str(reason or "").strip().lower() in CLIP_FINISH_REASONS


def is_clipped(result: Any, max_tokens: int) -> bool:
    """True if generation hit the token cap (finish reason or token count)."""
    if _finish_reason_clipped(getattr(result, "finish_reason", None)):
        return True
    n = int(getattr(result, "num_tokens", 0) or 0)
    cap = int(max_tokens)
    return cap > 0 and n >= cap


def is_clipped_record(record: dict[str, Any], max_tokens: int | None = None) -> bool:
    """True if a stored search row was length-capped or explicitly discarded."""
    if bool(record.get("discarded")) or bool(record.get("clipped")):
        return True
    if _finish_reason_clipped(record.get("finish_reason")):
        return True
    if max_tokens is None:
        return False
    n = int(record.get("generated_tokens") or 0)
    cap = int(max_tokens)
    return cap > 0 and n >= cap
