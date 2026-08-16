from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.data.build_training_dataset import build_training_rows, write_training_dataset
from offline_search.data.select_trajectories import SelectionCaps
from offline_search.eval.compare_models import write_comparison_table
from offline_search.runtime import load_experiment
from offline_search.search.resume import load_jsonl

OBJECTIVES = ("successful_sft", "positive_only", "binary_signed", "graded_signed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize baseline training sets from one search dump.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--search-jsonl", default=None)
    args = parser.parse_args()
    cfg = load_experiment(*args.config)
    search_jsonl = Path(args.search_jsonl or Path(cfg.output_dir) / "search" / "search_results.jsonl")
    records = load_jsonl(search_jsonl)
    from offline_search.data.build_training_dataset import CharTokenizer

    tokenizer = CharTokenizer()
    caps = SelectionCaps(
        max_correct_per_problem=cfg.selection.max_correct_per_problem,
        max_near_correct_per_problem=cfg.selection.max_near_correct_per_problem,
        max_hard_negatives_per_problem=cfg.selection.max_hard_negatives_per_problem,
        max_low_reward_negatives_per_problem=cfg.selection.max_low_reward_negatives_per_problem,
    )
    summary = {}
    for objective in OBJECTIVES:
        rows = build_training_rows(
            records,
            tokenizer=tokenizer,
            caps=caps,
            objective=objective,
            entropy_threshold=cfg.entropy.threshold,
            entropy_scale=cfg.entropy.scale,
            entropy_mode=cfg.entropy.mode,
        )
        out = Path(cfg.output_dir) / "baselines" / objective
        write_training_dataset(rows, out)
        summary[objective] = len(rows)
        print(f"{objective}: {len(rows)} rows -> {out}")
    (Path(cfg.output_dir) / "baselines" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_comparison_table({}, Path(cfg.output_dir) / "baselines" / "empty_table.json")


if __name__ == "__main__":
    main()
