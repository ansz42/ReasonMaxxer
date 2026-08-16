from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run(script: str, configs: list[str], env: dict[str, str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    for path in configs:
        cmd.extend(["--config", path])
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run search -> dataset -> train -> eval.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    if not env.get("WANDB_RUN_ID"):
        try:
            import wandb

            env["WANDB_RUN_ID"] = wandb.util.generate_id()
            env["WANDB_RESUME"] = "allow"
            print(f"wandb run id {env['WANDB_RUN_ID']}")
        except Exception:
            pass

    t0 = time.perf_counter()
    _run("01_search.py", args.config, env)
    _run("02_build_dataset.py", args.config, env)
    if not args.skip_train:
        _run("03_train.py", args.config, env)
        _run("04_eval.py", args.config, env)
    print(f"iteration_wall_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
