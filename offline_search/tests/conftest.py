from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    out = tmp_path / "outputs"
    out.mkdir()
    return out


def import_torch_or_skip():
    try:
        import torch

        torch.zeros(1)
        return torch
    except Exception as exc:  # DLL / CPU-build failures must skip, not error
        pytest.skip(f"torch unavailable: {exc}")
