from __future__ import annotations

import hashlib
from typing import Any


def stable_seed(*parts: Any, base: int = 42) -> int:
    payload = "::".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (int(base) + int(digest[:8], 16)) % (2**31 - 1)
