from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from offline_search.eval.pass_at_k import pass_at_k
from offline_search.search.generate import GenerationBackend
from offline_search.search.search_runner import Problem
from offline_search.scoring.base import RolloutScorer
from offline_search.utils.io import write_json
from offline_search.utils.seeds import stable_seed


def evaluate_backend(
    problems: Sequence[Problem],
    backend: GenerationBackend,
    scorer: RolloutScorer,
    *,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int = 42,
    ks: Sequence[int] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    ks = list(ks or [1, 4, 16])
    per_problem: list[dict[str, Any]] = []
    all_flags: list[bool] = []

    for problem in problems:
        flags: list[bool] = []
        rewards: list[float] = []
        responses: list[str] = []
        for i in range(int(n_samples)):
            sample_seed = stable_seed("eval", problem.problem_id, i, base=seed)
            outputs = backend.generate(
                [problem.prompt],
                temperature=temperature,
                top_p=top_p,
                n=1,
                max_tokens=max_tokens,
                seed=sample_seed,
            )
            text = outputs[0][0].text
            score = scorer.score_rollout(problem.prompt, text, problem.reference_answer)
            flags.append(bool(score.is_correct))
            rewards.append(float(score.reward))
            responses.append(text)
        n = len(flags)
        c = sum(1 for f in flags if f)
        metrics = {f"pass@{k}": pass_at_k(n, c, int(k)) for k in ks}
        per_problem.append(
            {
                "problem_id": problem.problem_id,
                "n": n,
                "n_correct": c,
                "mean_reward": sum(rewards) / max(1, n),
                "metrics": metrics,
                "responses": responses,
            }
        )
        all_flags.extend(flags)

    n_all = len(all_flags)
    c_all = sum(1 for f in all_flags if f)
    # Macro-average pass@k across problems (primary reported table).
    macro: dict[str, float] = {}
    for k in ks:
        values = [row["metrics"][f"pass@{k}"] for row in per_problem]
        macro[f"pass@{k}"] = sum(values) / max(1, len(values))

    result = {
        "macro": macro,
        "micro_correct_rate": (c_all / n_all) if n_all else 0.0,
        "num_problems": len(list(problems)),
        "n_samples": int(n_samples),
        "per_problem": per_problem,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def compare_eval_tables(named_results: dict[str, dict[str, Any]], ks: Sequence[int]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for name, result in named_results.items():
        row: dict[str, Any] = {"method": name}
        macro = result.get("macro", {})
        for k in ks:
            row[f"pass@{k}"] = macro.get(f"pass@{k}")
        table.append(row)
    return table
