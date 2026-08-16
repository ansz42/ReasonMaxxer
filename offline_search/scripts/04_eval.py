from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.eval.generate_eval import evaluate_backend
from offline_search.runtime import load_experiment, load_generation_stack, load_problems, make_scorer
from offline_search.utils.tracking import finish, gpu_snapshot, init_from_config, log as wandb_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pass@k for the current generation backend.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--label", default="model")
    args = parser.parse_args()

    cfg = load_experiment(*args.config)
    problems = load_problems(cfg)
    init_from_config(cfg, "eval")
    adapter = Path(cfg.output_dir) / "train" / "adapter"
    wandb_log({"stage/eval_start": 1, "eval/adapter": int(adapter.exists()), **gpu_snapshot()})
    _, _, backend = load_generation_stack(cfg, adapter_path=adapter if adapter.exists() else None)
    wandb_log({"stage/eval_model_loaded": 1, **gpu_snapshot()})
    output = Path(args.output or Path(cfg.output_dir) / "eval" / f"{args.label}.json")
    report = evaluate_backend(
        problems,
        backend,
        make_scorer(),
        n_samples=cfg.evaluation.n_samples,
        temperature=cfg.evaluation.temperature,
        top_p=cfg.evaluation.top_p,
        max_tokens=cfg.evaluation.max_tokens,
        seed=cfg.evaluation.seed,
        ks=cfg.evaluation.pass_k,
        output_path=output,
    )
    payload = {f"eval/{k}": v for k, v in report.get("macro", {}).items()}
    payload["eval/micro_correct_rate"] = report.get("micro_correct_rate")
    payload.update(gpu_snapshot())
    wandb_log(payload)
    finish()
    print(f"{args.label} macro: {report['macro']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
