from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.runtime import load_experiment, load_generation_stack, load_problems, make_scorer, search_settings_from_config
from offline_search.search.search_runner import run_search
from offline_search.utils.tracking import finish, gpu_snapshot, init_from_config, log as wandb_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive multi-config offline search.")
    parser.add_argument("--config", action="append", required=True, help="YAML config path (repeatable, later files override)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_experiment(*args.config)
    output_dir = Path(args.output_dir or cfg.output_dir) / "search"
    problems = load_problems(cfg)
    init_from_config(cfg, "search")
    wandb_log({"stage/search_start": 1, **gpu_snapshot()})
    _, _, backend = load_generation_stack(cfg)
    wandb_log({"stage/search_model_loaded": 1, **gpu_snapshot()})

    n = {"i": 0}

    def on_record(record, accounting) -> None:
        n["i"] += 1
        if n["i"] == 1 or n["i"] % 8 == 0:
            wandb_log(
                {
                    "search/rollouts": n["i"],
                    "search/tokens": accounting.generated_tokens,
                    "search/reward": float(record.get("reward", 0.0)),
                    "search/is_correct": int(bool(record.get("is_correct"))),
                    **gpu_snapshot(),
                }
            )

    result = run_search(
        problems,
        cfg.search.sampling_configs(),
        backend,
        make_scorer(),
        output_dir,
        search_settings_from_config(cfg),
        on_record=on_record,
    )
    records = result["records"]
    n_correct = sum(1 for r in records if r.get("is_correct"))
    mean_reward = (sum(float(r.get("reward", 0.0)) for r in records) / len(records)) if records else 0.0
    wandb_log(
        {
            "search/final_rollouts": len(records),
            "search/final_correct": n_correct,
            "search/final_correct_rate": (n_correct / len(records)) if records else 0.0,
            "search/final_mean_reward": mean_reward,
            "search/final_tokens": result["accounting"].get("generated_tokens", 0),
            "search/wall_time_s": result["accounting"].get("search_wall_time_s", 0.0),
            "search/tokens_per_s": result["accounting"].get("generated_tokens_per_s", 0.0),
            **gpu_snapshot(),
        }
    )
    finish()
    print(f"wrote {len(records)} rollouts to {result['parquet_path']}")
    print(f"generated_tokens={result['accounting']['generated_tokens']}")
    print(f"correct={n_correct}/{len(records)} mean_reward={mean_reward:.3f}")


if __name__ == "__main__":
    main()
