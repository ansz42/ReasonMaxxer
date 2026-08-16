from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_search.runtime import load_experiment
from offline_search.training.lora import load_unsloth_lora, save_lora
from offline_search.training.trainer import TrainSettings, train_signed_entropy
from offline_search.utils.io import records_from_parquet
from offline_search.utils.tracking import finish, gpu_snapshot, init_from_config, log as wandb_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with the signed entropy-weighted objective.")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_experiment(*args.config)
    dataset_path = Path(args.dataset or Path(cfg.output_dir) / "dataset" / "train_entropy.parquet")
    output_dir = Path(args.output_dir or Path(cfg.output_dir) / "train")
    rows = records_from_parquet(dataset_path)
    if not rows:
        # Fallback used by the CPU synthetic pack.
        json_fallback = dataset_path.with_suffix(".json")
        if json_fallback.exists():
            rows = json.loads(json_fallback.read_text(encoding="utf-8"))
        else:
            raise SystemExit(f"No training rows in {dataset_path}")

    t = cfg.training
    if t.backend != "unsloth":
        raise SystemExit("This test-pack trainer expects training.backend=unsloth")

    init_from_config(cfg, "train")
    wandb_log({"stage/train_start": 1, "train/num_rows": len(rows), **gpu_snapshot()})

    model, tokenizer = load_unsloth_lora(
        cfg.model.name,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        rank=t.lora_rank,
        alpha=t.lora_alpha,
        dropout=t.lora_dropout,
        target_modules=t.target_modules,
        seed=t.seed,
    )
    del tokenizer
    try:
        from unsloth import FastLanguageModel

        FastLanguageModel.for_training(model)
    except Exception:
        pass

    settings = TrainSettings(
        learning_rate=t.learning_rate,
        batch_size=t.batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        epochs=t.epochs,
        max_steps=t.max_steps,
        max_grad_norm=t.max_grad_norm,
        seed=t.seed,
        logging_steps=t.logging_steps,
        kl_coef=t.kl_coef,
    )
    metrics = train_signed_entropy(model, rows, settings=settings, output_dir=output_dir)
    save_lora(model, str(output_dir / "adapter"))
    acc = metrics.get("accounting", {})
    losses = [float(x["loss"]) for x in metrics.get("logs", [])]
    wandb_log(
        {
            "train/steps": metrics.get("steps", 0),
            "train/wall_time_s": acc.get("training_wall_time_s", 0.0),
            "train/tokens_total": acc.get("training_tokens", 0),
            "train/loss_first": losses[0] if losses else None,
            "train/loss_last": losses[-1] if losses else None,
            "train/loss_mean": (sum(losses) / len(losses)) if losses else None,
            **gpu_snapshot(),
        }
    )
    finish()
    print(f"trained {metrics['steps']} steps; adapter at {output_dir / 'adapter'}")
    if losses:
        print(f"loss first={losses[0]:.4f} last={losses[-1]:.4f} mean={sum(losses)/len(losses):.4f}")
    print(f"train_wall_s={acc.get('training_wall_time_s')} tokens={acc.get('training_tokens')}")


if __name__ == "__main__":
    main()
