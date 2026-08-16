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
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    max_steps: int | None = None
    max_grad_norm: float = 1.0
    seed: int = 42
    logging_steps: int = 5
    kl_coef: float = 0.0


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
    if not rows:
        raise ValueError("No training rows")
    _set_seed(settings.seed)
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.learning_rate))

    accounting = SearchAccounting()
    accounting.start_timer()
    logs: list[dict[str, Any]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.perf_counter()

    max_steps = settings.max_steps
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            token_count = int(batch["response_mask"].sum().item())
            accounting.training_tokens += token_count
            accounting.training_steps += 1
            global_step += 1
            raw_loss = float(loss.detach().cpu()) * max(1, int(settings.gradient_accumulation_steps))
            if global_step % int(settings.logging_steps) == 0 or global_step == 1:
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": raw_loss,
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
                        "train/tokens": token_count,
                        "train/advantage_mean": row["advantage_mean"],
                        "train/epoch": epoch,
                        **{k.replace("gpu/", "train/gpu_"): v for k, v in gpu_snapshot().items() if isinstance(v, (int, float))},
                    }
                )
            if max_steps is not None and global_step >= int(max_steps):
                break
        if max_steps is not None and global_step >= int(max_steps):
            break

    accounting.stop_train_timer()
    accounting.training_wall_time_s = time.perf_counter() - t0
    result = {
        "steps": global_step,
        "logs": logs,
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
