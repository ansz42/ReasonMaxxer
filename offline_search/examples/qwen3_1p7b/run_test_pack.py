#!/usr/bin/env python
"""Qwen3-1.7B test pack entrypoint.

Modes:
  check      Print environment status (torch / CUDA / unsloth).
  unit       Run the CPU unit suite in this repo.
  synthetic  Closed-loop pipeline with a scripted generator + tiny LM (no GPU).
  smoke      Real Qwen3-1.7B Unsloth LoRA loop using configs/test_pack_qwen3_1p7b.yaml.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TEST_PACK_CONFIG = ROOT / "configs" / "test_pack_qwen3_1p7b.yaml"


def _py() -> str:
    return sys.executable


def env_report() -> dict:
    report: dict = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": None,
        "cuda": False,
        "gpu": None,
        "unsloth": bool(importlib.util.find_spec("unsloth")),
        "transformers": bool(importlib.util.find_spec("transformers")),
        "peft": bool(importlib.util.find_spec("peft")),
        "ready_for_smoke": False,
    }
    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["torch_error"] = str(exc)
    report["ready_for_smoke"] = bool(report["unsloth"] and report["cuda"] and report["torch"])
    return report


def run_unit() -> int:
    return subprocess.call([_py(), "-m", "pytest", str(ROOT / "tests")])


def run_synthetic() -> int:
    return subprocess.call(
        [_py(), "-m", "pytest", str(ROOT / "tests" / "test_pipeline_synthetic.py"), "-q"]
    )


def run_smoke() -> int:
    report = env_report()
    print(json.dumps(report, indent=2))
    if not report["ready_for_smoke"]:
        print(
            "\nSmoke mode needs CUDA + Unsloth in this interpreter.\n"
            "The unit/synthetic pack is ready now. When the GPU env is set up:\n"
            "  pip install -r requirements.txt\n"
            "  python examples/qwen3_1p7b/run_test_pack.py --mode smoke\n"
        )
        return 2
    cmd = [
        _py(),
        str(ROOT / "scripts" / "run_iteration.py"),
        "--config",
        str(TEST_PACK_CONFIG),
    ]
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Qwen3-1.7B LoRA / Unsloth test pack.")
    parser.add_argument(
        "--mode",
        choices=["check", "unit", "synthetic", "smoke"],
        default="unit",
    )
    args = parser.parse_args()
    if args.mode == "check":
        print(json.dumps(env_report(), indent=2))
        return
    if args.mode == "unit":
        raise SystemExit(run_unit())
    if args.mode == "synthetic":
        raise SystemExit(run_synthetic())
    raise SystemExit(run_smoke())


if __name__ == "__main__":
    main()
