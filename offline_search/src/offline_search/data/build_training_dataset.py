from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from offline_search.data.advantages import attach_advantages, per_problem_advantages
from offline_search.data.compute_entropy import EntropyFn, attach_entropy_and_weights, uniform_entropy_fn
from offline_search.data.select_trajectories import SelectionCaps, select_trajectories
from offline_search.utils.io import records_to_parquet, write_json


VALID_OBJECTIVES = {
    "graded_signed",
    "binary_signed",
    "positive_only",
    "successful_sft",
}


def _binary_reward(row: dict[str, Any]) -> float:
    return 1.0 if row.get("is_correct") else 0.0


def apply_objective_rewards(rows: Sequence[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    if objective not in VALID_OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    out = [dict(r) for r in rows]
    if objective == "binary_signed":
        for row in out:
            row["reward"] = _binary_reward(row)
    return out


def apply_objective_advantages(rows: Sequence[dict[str, Any]], objective: str) -> list[dict[str, Any]]:
    if objective == "successful_sft":
        kept = [dict(r) for r in rows if r.get("is_correct")]
        for row in kept:
            row["advantage"] = 1.0
        return kept
    attached = attach_advantages(rows)
    if objective == "positive_only":
        for row in attached:
            row["advantage"] = max(0.0, float(row.get("advantage", 0.0)))
    return attached


class CharTokenizer:
    """Tiny deterministic tokenizer used by unit tests and the synthetic pack."""

    def encode(self, text: str) -> list[int]:
        if not text:
            return [1]
        return [1] + [(ord(ch) % 97) + 2 for ch in text[:255]]

    def decode(self, ids: Sequence[int]) -> str:
        chars: list[str] = []
        for i in ids:
            if int(i) <= 1:
                continue
            chars.append(chr(((int(i) - 2) % 97) + 32))
        return "".join(chars)


def tokenize_pair(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], int]:
    if hasattr(tokenizer, "encode") and not hasattr(tokenizer, "apply_chat_template"):
        prompt_ids = [int(x) for x in tokenizer.encode(prompt)]
        response_ids = [int(x) for x in tokenizer.encode(response)]
        if response_ids and prompt_ids and response_ids[: len(prompt_ids)] == prompt_ids:
            ids = response_ids
        else:
            ids = prompt_ids + response_ids
        return ids, len(prompt_ids)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
    if isinstance(prompt_ids, list) and prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
        full_ids = full_ids[0]
    return [int(x) for x in full_ids], int(len(prompt_ids))


def build_training_rows(
    search_records: Sequence[dict[str, Any]],
    *,
    tokenizer: Any,
    entropy_fn: EntropyFn | None = None,
    caps: SelectionCaps | None = None,
    objective: str = "graded_signed",
    entropy_threshold: float = 0.8,
    entropy_scale: float = 0.25,
    entropy_mode: str = "sigmoid",
) -> list[dict[str, Any]]:
    selected = select_trajectories(search_records, caps or SelectionCaps())
    selected = apply_objective_rewards(selected, objective)
    selected = apply_objective_advantages(selected, objective)
    entropy = entropy_fn or (lambda ids: uniform_entropy_fn(ids, 1.0))

    rows: list[dict[str, Any]] = []
    for rec in selected:
        ids, prompt_len = tokenize_pair(tokenizer, str(rec.get("prompt", "")), str(rec.get("response", "")))
        packed = attach_entropy_and_weights(
            ids,
            prompt_len,
            entropy,
            threshold=entropy_threshold,
            scale=entropy_scale,
            mode=entropy_mode,
        )
        if objective == "successful_sft":
            packed["token_weight"] = [float(m) for m in packed["response_mask"]]
        rows.append(
            {
                "problem_id": rec.get("problem_id"),
                "prompt": rec.get("prompt"),
                "response": rec.get("response"),
                "reward": float(rec.get("reward", 0.0)),
                "advantage": float(rec.get("advantage", 0.0)),
                "sampling_config_id": rec.get("sampling_config_id"),
                "is_correct": bool(rec.get("is_correct", False)),
                "near_correct": bool(rec.get("near_correct", False)),
                **packed,
            }
        )
    return rows


def write_training_dataset(rows: Sequence[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parquet_path = output / "train_entropy.parquet"
    json_path = output / "train_entropy.json"
    records_to_parquet(parquet_path, rows)
    write_json(json_path, list(rows))
    stats = {
        "num_rows": len(rows),
        "num_problems": len({str(r.get("problem_id")) for r in rows}),
        "num_positive": sum(1 for r in rows if float(r.get("advantage", 0.0)) > 0),
        "num_negative": sum(1 for r in rows if float(r.get("advantage", 0.0)) < 0),
        "mean_reward": (sum(float(r.get("reward", 0.0)) for r in rows) / len(rows)) if rows else 0.0,
    }
    write_json(output / "dataset_stats.json", stats)
    return {"parquet": str(parquet_path), "json": str(json_path)}


# Re-export for callers that only imported this module.
__all__ = [
    "CharTokenizer",
    "VALID_OBJECTIVES",
    "build_training_rows",
    "per_problem_advantages",
    "write_training_dataset",
]
