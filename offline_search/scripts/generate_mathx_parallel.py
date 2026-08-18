#!/usr/bin/env python
"""Generate n answers per MathX problem in one vLLM batched pass (20k total)."""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.runtime import load_experiment, load_generation_stack, load_problems, make_scorer
from offline_search.search.clipping import is_clipped
from offline_search.search.resume import append_jsonl, load_jsonl
from offline_search.utils.accounting import SearchAccounting
from offline_search.utils.io import records_to_parquet
from offline_search.utils.seeds import stable_seed
from offline_search.utils.tracking import (
    finish,
    gpu_snapshot,
    init_from_config,
    log as wandb_log,
    search_progress_metrics,
    should_log_search_progress,
)

CONFIG_ID = "t0.85_p0.95"
TEMPERATURE = 0.85
TOP_P = 0.95


def _chunks(items, size: int):
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _record(problem, result, scorer, *, sample_index: int, seed: int, max_tokens: int) -> dict:
    score = scorer.score_rollout(problem.prompt, result.text, problem.reference_answer)
    clipped = is_clipped(result, max_tokens)
    record = {
        "problem_id": problem.problem_id,
        "prompt": problem.prompt,
        "reference_answer": problem.reference_answer,
        "sampling_config_id": CONFIG_ID,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": None,
        "repetition_penalty": 1.0,
        "seed": int(seed),
        "sample_index": int(sample_index),
        "response": result.text,
        "reward": float(score.reward),
        "is_correct": bool(score.is_correct),
        "near_correct": bool(score.near_correct),
        "generated_tokens": int(result.num_tokens),
        "finish_reason": result.finish_reason,
        "clipped": bool(clipped),
        "clip_retried": False,
        "discarded": bool(clipped),
        "metadata": score.metadata,
    }
    rendered = (result.extra or {}).get("rendered_prompt")
    if rendered is not None:
        record["rendered_prompt"] = rendered
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Batched vLLM generation: 2000 problems x 10 answers.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--n", type=int, default=10, help="Answers per problem.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_experiment(*args.config)
    n_per = max(1, int(args.n))
    output_dir = Path(args.output_dir or Path(cfg.output_dir) / "search")
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "search_results.jsonl"
    parquet_path = output_dir / "search_results.parquet"

    problems = load_problems(cfg)
    existing = load_jsonl(jsonl_path)
    done: dict[str, int] = defaultdict(int)
    for rec in existing:
        done[str(rec.get("problem_id"))] += 1
    pending = [p for p in problems if done[p.problem_id] < n_per]
    target = len(problems) * n_per
    already = sum(min(done[p.problem_id], n_per) for p in problems)

    init_from_config(cfg, "search")
    wandb_log(
        {
            "stage/search_start": 1,
            "search/target_rollouts": target,
            "search/resumed_rollouts": already,
            "search/pending_problems": len(pending),
        },
        step=already,
    )

    state = {
        "i": already,
        "tokens": int(sum(int(r.get("generated_tokens") or 0) for r in existing)),
        "reward_sum": float(sum(float(r.get("reward") or 0.0) for r in existing)),
        "correct": int(sum(1 for r in existing if r.get("is_correct"))),
    }
    print(
        f"mathx generate resume {already}/{target} pending_problems={len(pending)} "
        f"n={n_per} batch={cfg.search.generation_batch_size}",
        flush=True,
    )

    if not pending:
        records_to_parquet(parquet_path, existing)
        print(f"already complete: {already} rollouts -> {parquet_path}", flush=True)
        finish()
        return

    _, _, backend = load_generation_stack(cfg)
    scorer = make_scorer()
    wandb_log({"stage/search_model_loaded": 1, **gpu_snapshot()}, step=already)
    accounting = SearchAccounting()
    accounting.start_timer()
    max_tokens = int(cfg.search.max_tokens)
    batch_size = max(1, int(cfg.search.generation_batch_size))
    t0 = time.time()

    def on_record(record) -> None:
        state["i"] += 1
        state["tokens"] += int(record.get("generated_tokens") or 0)
        state["reward_sum"] += float(record.get("reward") or 0.0)
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
        elapsed = max(1e-6, time.time() - t0)
        rate = (step - already) / elapsed
        print(
            f"search step {step}/{target} "
            f"reward={payload['search/reward']:.3f} "
            f"mean_reward={payload['search/reward_mean']:.3f} "
            f"correct={state['correct']}/{step} ({payload['search/correct_rate']:.3f}) "
            f"{rate:.2f} ans/s",
            flush=True,
        )

    for chunk in _chunks(pending, batch_size):
        remaining = [n_per - done[p.problem_id] for p in chunk]
        # Same n for the whole vLLM call; pending problems almost always need full n.
        n_this = max(remaining)
        seeds = [stable_seed(p.problem_id, CONFIG_ID, 0, base=cfg.search.seed) for p in chunk]
        outputs = backend.generate(
            [p.prompt for p in chunk],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            n=n_this,
            max_tokens=max_tokens,
            seed=int(seeds[0]),
            seeds=seeds,
        )
        if len(outputs) != len(chunk):
            raise RuntimeError(f"backend returned {len(outputs)} rows for {len(chunk)} prompts")
        new_records = []
        for problem, rows, need in zip(chunk, outputs, remaining):
            start_i = done[problem.problem_id]
            for offset, result in enumerate(rows[:need]):
                sample_index = start_i + offset
                seed = stable_seed(problem.problem_id, CONFIG_ID, sample_index, base=cfg.search.seed)
                record = _record(problem, result, scorer, sample_index=sample_index, seed=seed, max_tokens=max_tokens)
                new_records.append(record)
                accounting.add_rollout(int(result.num_tokens))
                on_record(record)
            done[problem.problem_id] += min(len(rows), need)
        append_jsonl(jsonl_path, new_records)

    accounting.stop_search_timer()
    final = load_jsonl(jsonl_path)
    records_to_parquet(parquet_path, final)
    accounting.extra["num_problems"] = len(problems)
    accounting.extra["n_per_problem"] = n_per
    accounting.extra["sampling_config_id"] = CONFIG_ID
    accounting.write(output_dir / "accounting.json")
    n_correct = sum(1 for r in final if r.get("is_correct"))
    mean_reward = (sum(float(r.get("reward") or 0.0) for r in final) / len(final)) if final else 0.0
    wandb_log(
        {
            "search/step": len(final),
            "search/final_rollouts": len(final),
            "search/final_correct": n_correct,
            "search/final_correct_rate": (n_correct / len(final)) if final else 0.0,
            "search/final_mean_reward": mean_reward,
            "search/final_tokens": accounting.to_dict().get("generated_tokens", 0),
            "search/wall_time_s": accounting.to_dict().get("search_wall_time_s", 0.0),
            **gpu_snapshot(),
        },
        step=len(final),
    )
    finish()
    print(f"wrote {len(final)} rollouts to {parquet_path}", flush=True)
    print(f"generated_tokens={accounting.to_dict().get('generated_tokens')} correct={n_correct}/{len(final)} mean_reward={mean_reward:.3f}", flush=True)


if __name__ == "__main__":
    main()
