from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.data.build_training_dataset import build_training_rows, write_training_dataset
from offline_search.data.compute_entropy import model_entropy_fn, uniform_entropy_fn
from offline_search.data.select_trajectories import SelectionCaps
from offline_search.runtime import load_experiment, load_torch_stack
from offline_search.search.resume import load_jsonl
from offline_search.utils.tracking import finish, gpu_snapshot, init_from_config, log as wandb_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Select trajectories, compute entropy, write train_entropy.parquet.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--search-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-model-entropy", action="store_true", help="Use uniform entropy (CPU / debug).")
    args = parser.parse_args()

    cfg = load_experiment(*args.config)
    search_jsonl = Path(args.search_jsonl or Path(cfg.output_dir) / "search" / "search_results.jsonl")
    output_dir = Path(args.output_dir or Path(cfg.output_dir) / "dataset")
    records = load_jsonl(search_jsonl)
    if not records:
        raise SystemExit(f"No search records in {search_jsonl}")
    init_from_config(cfg, "dataset")
    wandb_log({"stage/dataset_start": 1, "dataset/search_records": len(records), **gpu_snapshot()})

    tokenizer = None
    entropy_fn = None
    if args.skip_model_entropy:
        from offline_search.data.build_training_dataset import CharTokenizer

        tokenizer = CharTokenizer()
        entropy_fn = lambda ids: uniform_entropy_fn(ids, 1.0)
    else:
        model, tokenizer = load_torch_stack(cfg)
        device = None
        try:
            device = str(next(model.parameters()).device)
        except Exception:
            device = None
        entropy_fn = model_entropy_fn(model, device=device)

    rows = build_training_rows(
        records,
        tokenizer=tokenizer,
        entropy_fn=entropy_fn,
        caps=SelectionCaps(
            max_correct_per_problem=cfg.selection.max_correct_per_problem,
            max_near_correct_per_problem=cfg.selection.max_near_correct_per_problem,
            max_hard_negatives_per_problem=cfg.selection.max_hard_negatives_per_problem,
            max_low_reward_negatives_per_problem=cfg.selection.max_low_reward_negatives_per_problem,
            drop_clipped=cfg.selection.drop_clipped,
            max_generated_tokens=cfg.selection.max_generated_tokens or cfg.search.max_tokens,
        ),
        objective=cfg.training.objective,
        entropy_threshold=cfg.entropy.threshold,
        entropy_scale=cfg.entropy.scale,
        entropy_mode=cfg.entropy.mode,
        enable_thinking=cfg.model.enable_thinking,
    )
    paths = write_training_dataset(rows, output_dir)
    pos = sum(1 for r in rows if float(r.get("advantage", 0.0)) > 0)
    neg = sum(1 for r in rows if float(r.get("advantage", 0.0)) < 0)
    informative = sum(1 for r in rows if abs(float(r.get("advantage", 0.0))) > 1e-8)
    wandb_log(
        {
            "dataset/num_rows": len(rows),
            "dataset/num_positive": pos,
            "dataset/num_negative": neg,
            "dataset/num_informative": informative,
            "dataset/num_zero_advantage": len(rows) - informative,
            "dataset/mean_reward": (sum(float(r.get("reward", 0.0)) for r in rows) / len(rows)) if rows else 0.0,
            **gpu_snapshot(),
        }
    )
    finish()
    print(f"wrote {len(rows)} training rows ({informative} informative) to {paths['parquet']}")


if __name__ == "__main__":
    main()
