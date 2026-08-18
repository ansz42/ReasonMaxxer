from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from offline_search.training.loss import causal_token_logprobs, decision_loss
from offline_search.utils.accounting import SearchAccounting
from offline_search.utils.io import write_json
from offline_search.utils.tracking import gpu_snapshot, log as wandb_log


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


@dataclass
class TrainSettings:
    learning_rate: float = 2.0e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    epochs: int = 1
    max_steps: int | None = None
    max_grad_norm: float = 0.1
    seed: int = 42
    logging_steps: int = 5
    save_steps: int | None = 100
    kl_coef: float = 0.0
    drop_zero_advantage: bool = True
    min_abs_advantage: float = 1e-8
    cover_all_informative: bool = True
    warmup_steps: int = 0
    warmup_ratio: float = 0.0


def should_save_checkpoint(step: int, save_steps: int | None) -> bool:
    if save_steps is None:
        return False
    interval = int(save_steps)
    return interval > 0 and int(step) > 0 and int(step) % interval == 0


def checkpoint_dir(output_dir: str | Path, step: int) -> Path:
    return Path(output_dir) / f"checkpoint-{int(step)}"


def save_lora_checkpoint(model: Any, output_dir: str | Path, step: int) -> Path:
    dest = checkpoint_dir(output_dir, step)
    dest.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(dest)
    else:
        raise TypeError("Model does not support save_pretrained")
    return dest


def filter_informative_rows(
    rows: Sequence[dict[str, Any]],
    *,
    min_abs_advantage: float = 1e-8,
    drop_zero_advantage: bool = True,
) -> list[dict[str, Any]]:
    if not drop_zero_advantage:
        return list(rows)
    eps = float(min_abs_advantage)
    return [r for r in rows if abs(float(r.get("advantage", 0.0))) > eps]


def resolve_max_steps(n_rows: int, settings: TrainSettings) -> int | None:
    """Keep a configured cap, but never stop before one pass over informative rows."""
    batch = max(1, int(settings.batch_size))
    one_pass = (max(0, int(n_rows)) + batch - 1) // batch
    if settings.cover_all_informative and n_rows > 0:
        if settings.max_steps is None:
            return None
        return max(int(settings.max_steps), one_pass)
    return settings.max_steps


def scheduled_lr(base_lr: float, update_index: int, warmup_updates: int) -> float:
    """Linear warmup over optimizer updates, then hold at base_lr.

    ``update_index`` is 0 before the first optimizer step and increments after
    each ``optimizer.step()``.
    """
    lr = float(base_lr)
    warm = max(0, int(warmup_updates))
    if warm <= 0:
        return lr
    step = max(0, int(update_index))
    if step >= warm:
        return lr
    return lr * float(step) / float(warm)


def resolve_warmup_updates(n_rows: int, settings: TrainSettings) -> int:
    """Warmup length in optimizer updates (not micro-batches)."""
    if int(settings.warmup_steps) > 0:
        return max(0, int(settings.warmup_steps))
    ratio = float(settings.warmup_ratio)
    if ratio <= 0:
        return 0
    micro = resolve_max_steps(n_rows, settings)
    if micro is None:
        batch = max(1, int(settings.batch_size))
        micro = (max(0, int(n_rows)) + batch - 1) // batch
    accum = max(1, int(settings.gradient_accumulation_steps))
    updates = (max(0, int(micro)) + accum - 1) // accum
    return max(0, int(updates * ratio))


def _pad_batch(rows: Sequence[dict[str, Any]], pad_id: int = 0) -> dict[str, Any]:
    import torch

    max_len = max(len(r["input_ids"]) for r in rows)
    input_ids = []
    attention = []
    weights = []
    masks = []
    advantages = []
    for row in rows:
        ids = list(row["input_ids"])
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        w = list(row["token_weight"]) + [0.0] * pad
        m = list(row["response_mask"]) + [0] * pad
        weights.append(w)
        masks.append(m)
        advantages.append(float(row.get("advantage", 0.0)))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
        "token_weight": torch.tensor(weights, dtype=torch.float32),
        "response_mask": torch.tensor(masks, dtype=torch.float32),
        "advantage": torch.tensor(advantages, dtype=torch.float32),
    }


def _iter_batches(rows: Sequence[dict[str, Any]], batch_size: int, seed: int):
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [rows[i] for i in order[start : start + batch_size]]


def train_signed_entropy(
    model: Any,
    rows: Sequence[dict[str, Any]],
    *,
    settings: TrainSettings | None = None,
    output_dir: str | Path | None = None,
    pad_id: int = 0,
) -> dict[str, Any]:
    import torch

    settings = settings or TrainSettings()
    incoming = list(rows)
    if not incoming:
        raise ValueError("No training rows")
    rows = filter_informative_rows(
        incoming,
        min_abs_advantage=settings.min_abs_advantage,
        drop_zero_advantage=settings.drop_zero_advantage,
    )
    if not rows:
        raise ValueError(
            "No training rows with |advantage| > "
            f"{settings.min_abs_advantage}; all {len(incoming)} rows were no-ops"
        )
    _set_seed(settings.seed)
    device = next(model.parameters()).device
    model.train()
    base_lr = float(settings.learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    warmup_updates = resolve_warmup_updates(len(rows), settings)
    for group in optimizer.param_groups:
        group["lr"] = scheduled_lr(base_lr, 0, warmup_updates)

    accounting = SearchAccounting()
    accounting.start_timer()
    logs: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    global_step = 0
    optimizer_updates = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()

    max_steps = resolve_max_steps(len(rows), settings)
    for epoch in range(int(settings.epochs)):
        for batch_rows in _iter_batches(rows, settings.batch_size, settings.seed + epoch):
            batch = _pad_batch(batch_rows, pad_id=pad_id)
            input_ids = batch["input_ids"].to(device)
            attention = batch["attention_mask"].to(device)
            weights = (batch["token_weight"] * batch["response_mask"]).to(device)
            advantage = batch["advantage"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            logprobs = causal_token_logprobs(logits, input_ids)
            loss = decision_loss(logprobs, advantage, weights)
            loss = loss / max(1, int(settings.gradient_accumulation_steps))
            loss.backward()

            if (global_step + 1) % int(settings.gradient_accumulation_steps) == 0:
                optimizer_updates += 1
                lr = scheduled_lr(base_lr, optimizer_updates, warmup_updates)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            token_count = int(batch["response_mask"].sum().item())
            accounting.training_tokens += token_count
            accounting.training_steps += 1
            global_step += 1
            raw_loss = float(loss.detach().cpu()) * max(1, int(settings.gradient_accumulation_steps))
            current_lr = float(optimizer.param_groups[0]["lr"])
            if global_step % int(settings.logging_steps) == 0 or global_step == 1:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": raw_loss,
                    "lr": current_lr,
                    "tokens": token_count,
                    "advantage_mean": float(batch["advantage"].mean().item()),
                    "step_time_s": time.perf_counter() - t0,
                    **{k: v for k, v in gpu_snapshot().items() if isinstance(v, (int, float))},
                }
                logs.append(row)
                wandb_log(
                    {
                        "train/loss": raw_loss,
                        "train/step": global_step,
                        "train/lr": current_lr,
                        "train/tokens": token_count,
                        "train/advantage_mean": row["advantage_mean"],
                        "train/epoch": epoch,
                        **{k.replace("gpu/", "train/gpu_"): v for k, v in gpu_snapshot().items() if isinstance(v, (int, float))},
                    }
                )
            if output_dir is not None and should_save_checkpoint(global_step, settings.save_steps):
                dest = save_lora_checkpoint(model, output_dir, global_step)
                checkpoints.append(str(dest))
                wandb_log({"train/checkpoint_step": global_step})
                print(f"saved LoRA checkpoint to {dest}")
            if max_steps is not None and global_step >= int(max_steps):
                break
        if max_steps is not None and global_step >= int(max_steps):
            break

    accounting.stop_train_timer()
    accounting.training_wall_time_s = time.perf_counter() - t0
    result = {
        "steps": global_step,
        "logs": logs,
        "num_rows_in": len(incoming),
        "num_rows": len(rows),
        "num_rows_dropped": len(incoming) - len(rows),
        "max_steps_effective": max_steps,
        "warmup_updates": warmup_updates,
        "optimizer_updates": optimizer_updates,
        "checkpoints": checkpoints,
        "accounting": accounting.to_dict(),
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(out / "adapter")
        write_json(out / "train_metrics.json", result)
        accounting.write(out / "accounting.json")
    return result
