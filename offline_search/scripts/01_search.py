from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.runtime import load_experiment, load_generation_stack, load_problems, make_scorer, search_settings_from_config
from offline_search.search.resume import load_jsonl
from offline_search.search.search_runner import run_search
from offline_search.utils.tracking import (
    finish,
    gpu_snapshot,
    init_from_config,
    log as wandb_log,
    search_progress_metrics,
    should_log_search_progress,
    summarize_search_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive multi-config offline search.")
    parser.add_argument("--config", action="append", required=True, help="YAML config path (repeatable, later files override)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_experiment(*args.config)
    output_dir = Path(args.output_dir or cfg.output_dir) / "search"
    problems = load_problems(cfg)
    existing = summarize_search_records(load_jsonl(output_dir / "search_results.jsonl"))
    target = len(problems) * int(cfg.search.total_samples_per_problem)
    state = {
        "i": int(existing["n"]),
        "tokens": int(existing["tokens"]),
        "reward_sum": float(existing["reward_sum"]),
        "correct": int(existing["correct"]),
    }
    init_from_config(cfg, "search")
    wandb_log(
        {
            "stage/search_start": 1,
            "search/target_rollouts": target,
            "search/resumed_rollouts": state["i"],
        },
        step=0,
    )
    _, _, backend = load_generation_stack(cfg)
    wandb_log({"stage/search_model_loaded": 1, **gpu_snapshot()}, step=0)
    if state["i"] > 0:
        wandb_log(
            {
                "search/step": state["i"],
                "search/rollouts": state["i"],
                "search/tokens": state["tokens"],
                "search/reward_mean": existing["reward_mean"],
                "search/correct_rate": existing["correct_rate"],
                "search/resumed": 1,
                **gpu_snapshot(),
            },
            step=state["i"],
        )
        print(
            f"search resume at step {state['i']}/{target} "
            f"mean_reward={existing['reward_mean']:.3f} "
            f"correct={state['correct']}/{state['i']}",
            flush=True,
        )

    def on_record(record, accounting) -> None:
        del accounting
        state["i"] += 1
        state["tokens"] += int(record.get("generated_tokens", 0) or 0)
        state["reward_sum"] += float(record.get("reward", 0.0) or 0.0)
        state["correct"] += int(bool(record.get("is_correct")))
        step = int(state["i"])
        if not should_log_search_progress(step):
            return
        payload = search_progress_metrics(
            step,
            record,
            tokens=state["tokens"],
            reward_sum=state["reward_sum"],
            correct=state["correct"],
            extra=gpu_snapshot(),
        )
        wandb_log(payload, step=step)
        print(
            f"search step {step}/{target} "
            f"reward={payload['search/reward']:.3f} "
            f"mean_reward={payload['search/reward_mean']:.3f} "
            f"correct={state['correct']}/{step} ({payload['search/correct_rate']:.3f})",
            flush=True,
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
    final_step = len(records)
    wandb_log(
        {
            "search/step": final_step,
            "search/final_rollouts": final_step,
            "search/final_correct": n_correct,
            "search/final_correct_rate": (n_correct / final_step) if final_step else 0.0,
            "search/final_mean_reward": mean_reward,
            "search/final_tokens": result["accounting"].get("generated_tokens", 0),
            "search/wall_time_s": result["accounting"].get("search_wall_time_s", 0.0),
            "search/tokens_per_s": result["accounting"].get("generated_tokens_per_s", 0.0),
            **gpu_snapshot(),
        },
        step=final_step,
    )
    finish()
    print(f"wrote {len(records)} rollouts to {result['parquet_path']}")
    print(f"generated_tokens={result['accounting']['generated_tokens']}")
    print(f"correct={n_correct}/{len(records)} mean_reward={mean_reward:.3f}")


if __name__ == "__main__":
    main()
