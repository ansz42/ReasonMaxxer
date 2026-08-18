#!/usr/bin/env python
"""Evaluate a merged / Hub model on GSM8K and MATH-500 (greedy pass@1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.eval.benchmarks import BENCHMARK_LOADERS, BENCHMARK_SOURCES, load_benchmark_problems
from offline_search.eval.generate_eval import evaluate_backend
from offline_search.runtime import load_experiment
from offline_search.search.generate import VLLMBackend
from offline_search.scoring.math_verifier import MathVerifier
from offline_search.utils.io import write_json
from offline_search.utils.tracking import finish, gpu_snapshot, init_from_config, log as wandb_log


def _parse_benchmarks(raw: str | None, fallback: list[str]) -> list[str]:
    if not raw:
        return list(fallback)
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [name for name in names if name not in BENCHMARK_LOADERS]
    if unknown:
        raise SystemExit(f"Unknown benchmarks {unknown}. Known: {sorted(BENCHMARK_LOADERS)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy pass@1 math-benchmark harness.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--model", default=None, help="Local dir or Hub id. Default: config model.name.")
    parser.add_argument("--benchmarks", default=None, help="Comma list. Default: gsm8k,math500.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-benchmark cap.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label", default="math-test-maxx")
    args = parser.parse_args()

    cfg = load_experiment(*args.config)
    model_name = args.model or cfg.model.name
    names = _parse_benchmarks(args.benchmarks, list(cfg.raw.get("benchmarks") or ["gsm8k", "math500"]))
    output_dir = Path(args.output_dir or Path(cfg.output_dir) / "eval_harness")
    ev = cfg.evaluation
    n_samples = 1
    temperature = 0.0
    top_p = 1.0
    ks = [1]
    max_tokens = ev.max_tokens
    batch = ev.generation_batch_size or cfg.search.generation_batch_size

    init_from_config(cfg, "eval_harness")
    wandb_log(
        {
            "stage/eval_harness_start": 1,
            "eval_harness/model": model_name,
            "eval_harness/benchmarks": ",".join(names),
            **gpu_snapshot(),
        }
    )
    print(f"loading vLLM model {model_name}")
    backend = VLLMBackend(
        model_name=model_name,
        max_model_len=cfg.model.max_seq_length,
        enable_thinking=cfg.model.enable_thinking,
        gpu_memory_utilization=cfg.search.gpu_memory_utilization,
        enforce_eager=cfg.search.enforce_eager,
    )
    scorer = MathVerifier()
    summary: dict[str, object] = {
        "model": model_name,
        "label": args.label,
        "benchmarks": {},
    }
    for name in names:
        problems = load_benchmark_problems(name, prompt_style=cfg.model.prompt_style, limit=args.limit)
        print(f"eval {name}: {len(problems)} problems, greedy pass@1, max_tokens={max_tokens}, batch={batch}")
        report = evaluate_backend(
            problems,
            backend,
            scorer,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=ev.seed,
            ks=ks,
            output_path=output_dir / f"{args.label}_{name}.json",
            generation_batch_size=batch,
        )
        row = {
            "source": BENCHMARK_SOURCES.get(name, name),
            "num_problems": report["num_problems"],
            "pass@1": report["macro"].get("pass@1"),
            "micro_correct_rate": report["micro_correct_rate"],
        }
        summary["benchmarks"][name] = row
        wandb_log(
            {
                f"eval_harness/{name}/pass@1": row["pass@1"],
                f"eval_harness/{name}/micro": row["micro_correct_rate"],
                f"eval_harness/{name}/n": row["num_problems"],
            }
        )
        print(f"{name} pass@1={row['pass@1']:.4f} micro={row['micro_correct_rate']:.4f} n={row['num_problems']}")

    write_json(output_dir / f"{args.label}_summary.json", summary)
    wandb_log({"stage/eval_harness_done": 1, **gpu_snapshot()})
    finish()
    print(json.dumps(summary, indent=2))
    print(f"wrote {output_dir / f'{args.label}_summary.json'}")


if __name__ == "__main__":
    main()
