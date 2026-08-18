#!/usr/bin/env python
"""Merge the latest LoRA into the base model and upload it to the Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.runtime import load_experiment
from offline_search.training.merge import (
    merge_adapter,
    push_model_dir,
    resolve_hub_repo_id,
    resolve_latest_adapter,
    write_model_card,
)
from offline_search.utils.tracking import finish, gpu_snapshot, init_from_config, log as wandb_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge latest LoRA adapter and push math-test-maxx.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--adapter", default=None, help="Override adapter dir. Default: latest under output_dir/train.")
    parser.add_argument("--output-dir", default=None, help="Where to write merged 16-bit weights.")
    parser.add_argument("--name", default="math-test-maxx")
    parser.add_argument("--repo-id", default=None, help="Full Hub id, e.g. user/math-test-maxx.")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment(*args.config)
    train_dir = Path(cfg.output_dir) / "train"
    adapter = Path(args.adapter) if args.adapter else resolve_latest_adapter(train_dir)
    merged_dir = Path(args.output_dir or Path(cfg.output_dir) / "merged" / args.name)
    repo_id = None if args.no_push else resolve_hub_repo_id(name=args.name, repo_id=args.repo_id)

    init_from_config(cfg, "merge")
    wandb_log(
        {
            "stage/merge_start": 1,
            "merge/adapter": str(adapter),
            "merge/output": str(merged_dir),
            **gpu_snapshot(),
        }
    )
    print(f"merging adapter {adapter} into {cfg.model.name} -> {merged_dir}")
    merge_adapter(
        base_model=cfg.model.name,
        adapter_path=adapter,
        output_dir=merged_dir,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
    )
    write_model_card(
        merged_dir,
        repo_id=repo_id or args.name,
        base_model=cfg.model.name,
        adapter_path=adapter,
    )
    url = None
    if not args.no_push:
        assert repo_id is not None
        print(f"uploading {merged_dir} -> {repo_id}")
        url = push_model_dir(merged_dir, repo_id, private=args.private)
        print(f"hub {url}")
    wandb_log(
        {
            "stage/merge_done": 1,
            "merge/hub_repo": repo_id or "",
            **gpu_snapshot(),
        }
    )
    finish()
    print(f"merged_dir={merged_dir}")
    if url:
        print(f"repo={url}")


if __name__ == "__main__":
    main()
