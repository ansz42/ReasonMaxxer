#!/usr/bin/env python
"""Sample 300 random MATH-500 test problems for the Qwen2.5-3B test pack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.data.math500 import DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, sample_math500, write_records

DEFAULT_OUT = ROOT / "examples" / "qwen25_3b" / "fixtures" / "math500_300.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample HuggingFaceH4/MATH-500[test] into a problems JSON.")
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = sample_math500(n=int(args.n), seed=int(args.seed))
    path = write_records(args.out, payload)
    print(f"wrote {payload['meta']['num_records']} records to {path}")
    print(
        "dataset={dataset} split={split} seed={seed}".format(**payload["meta"])
    )


if __name__ == "__main__":
    main()
