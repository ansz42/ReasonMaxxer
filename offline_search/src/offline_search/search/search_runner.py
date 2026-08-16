from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any, Sequence

from offline_search.search.adaptive_allocator import allocate_remaining, config_score
from offline_search.search.generate import GenerationBackend
from offline_search.search.resume import JobKey, append_jsonl, completed_job_keys, filter_pending, job_key, load_jsonl
from offline_search.search.sampling_configs import SamplingConfig
from offline_search.scoring.base import RolloutScorer
from offline_search.utils.accounting import SearchAccounting
from offline_search.utils.io import records_to_parquet
from offline_search.utils.seeds import stable_seed


@dataclass
class Problem:
    problem_id: str
    prompt: str
    reference_answer: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchJob:
    problem_id: str
    prompt: str
    reference_answer: str | None
    sampling_config_id: str
    temperature: float
    top_p: float
    top_k: int | None
    repetition_penalty: float
    seed: int
    sample_index: int

    @property
    def key(self) -> JobKey:
        return job_key(self.problem_id, self.sampling_config_id, self.seed)


@dataclass
class SearchSettings:
    initial_samples_per_config: int = 16
    total_samples_per_problem: int = 256
    exploration_fraction: float = 0.20
    allocation_temperature: float = 0.5
    max_tokens: int = 1024
    seed: int = 42
    flush_every: int = 1


def _config_stats(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_cfg[str(rec["sampling_config_id"])].append(rec)
    stats: dict[str, dict[str, float]] = {}
    for cfg_id, rows in by_cfg.items():
        n = max(1, len(rows))
        correct = sum(1 for r in rows if r.get("is_correct"))
        near = sum(1 for r in rows if r.get("near_correct"))
        mean_reward = sum(float(r.get("reward", 0.0)) for r in rows) / n
        stats[cfg_id] = {
            "n": float(len(rows)),
            "correct_rate": correct / n,
            "near_correct_rate": near / n,
            "mean_reward": mean_reward,
            "score": config_score(correct / n, near / n, mean_reward),
        }
    return stats


def plan_initial_jobs(
    problems: Sequence[Problem],
    configs: Sequence[SamplingConfig],
    settings: SearchSettings,
) -> list[SearchJob]:
    jobs: list[SearchJob] = []
    for problem in problems:
        for config in configs:
            for sample_i in range(int(settings.initial_samples_per_config)):
                seed = stable_seed(problem.problem_id, config.config_id, sample_i, base=settings.seed)
                jobs.append(
                    SearchJob(
                        problem_id=problem.problem_id,
                        prompt=problem.prompt,
                        reference_answer=problem.reference_answer,
                        sampling_config_id=config.config_id,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        repetition_penalty=config.repetition_penalty,
                        seed=seed,
                        sample_index=sample_i,
                    )
                )
    return jobs


def plan_adaptive_jobs(
    problems: Sequence[Problem],
    configs: Sequence[SamplingConfig],
    existing: Sequence[dict[str, Any]],
    settings: SearchSettings,
) -> list[SearchJob]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in existing:
        by_problem[str(rec["problem_id"])].append(rec)

    config_by_id = {c.config_id: c for c in configs}
    jobs: list[SearchJob] = []
    for problem in problems:
        rows = by_problem.get(problem.problem_id, [])
        remaining = int(settings.total_samples_per_problem) - len(rows)
        if remaining <= 0:
            continue
        stats = _config_stats(rows)
        scores = [float(stats.get(c.config_id, {}).get("score", 0.0)) for c in configs]
        alloc = allocate_remaining(
            scores,
            remaining,
            exploration_fraction=settings.exploration_fraction,
            allocation_temperature=settings.allocation_temperature,
        )
        used_by_cfg: dict[str, int] = defaultdict(int)
        for rec in rows:
            used_by_cfg[str(rec["sampling_config_id"])] += 1
        for config, extra_n in zip(configs, alloc):
            start = used_by_cfg[config.config_id]
            for offset in range(int(extra_n)):
                sample_i = start + offset
                seed = stable_seed(problem.problem_id, config.config_id, sample_i, base=settings.seed)
                jobs.append(
                    SearchJob(
                        problem_id=problem.problem_id,
                        prompt=problem.prompt,
                        reference_answer=problem.reference_answer,
                        sampling_config_id=config.config_id,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        repetition_penalty=config.repetition_penalty,
                        seed=seed,
                        sample_index=sample_i,
                    )
                )
    return jobs


def _execute_jobs(
    jobs: Sequence[SearchJob],
    backend: GenerationBackend,
    scorer: RolloutScorer,
    *,
    settings: SearchSettings,
    jsonl_path: Path,
    accounting: SearchAccounting,
    on_record: Callable[[dict[str, Any], SearchAccounting], None] | None = None,
) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    pending = list(jobs)
    # Execute one completion per job so seeds stay unique and resume keys stay stable.
    for job in pending:
        outputs = backend.generate(
            [job.prompt],
            temperature=job.temperature,
            top_p=job.top_p,
            n=1,
            max_tokens=settings.max_tokens,
            seed=job.seed,
            top_k=job.top_k,
            repetition_penalty=job.repetition_penalty,
        )
        result = outputs[0][0]
        score = scorer.score_rollout(job.prompt, result.text, job.reference_answer)
        record = {
            "problem_id": job.problem_id,
            "prompt": job.prompt,
            "reference_answer": job.reference_answer,
            "sampling_config_id": job.sampling_config_id,
            "temperature": job.temperature,
            "top_p": job.top_p,
            "top_k": job.top_k,
            "repetition_penalty": job.repetition_penalty,
            "seed": job.seed,
            "sample_index": job.sample_index,
            "response": result.text,
            "reward": float(score.reward),
            "is_correct": bool(score.is_correct),
            "near_correct": bool(score.near_correct),
            "generated_tokens": int(result.num_tokens),
            "metadata": score.metadata,
        }
        rendered = (result.extra or {}).get("rendered_prompt")
        if rendered is not None:
            record["rendered_prompt"] = rendered
        append_jsonl(jsonl_path, [record])
        accounting.add_rollout(result.num_tokens)
        written.append(record)
        if on_record is not None:
            on_record(record, accounting)
    return written


def run_search(
    problems: Sequence[Problem],
    configs: Sequence[SamplingConfig],
    backend: GenerationBackend,
    scorer: RolloutScorer,
    output_dir: str | Path,
    settings: SearchSettings | None = None,
    on_record: Callable[[dict[str, Any], SearchAccounting], None] | None = None,
) -> dict[str, Any]:
    settings = settings or SearchSettings()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    jsonl_path = output / "search_results.jsonl"
    parquet_path = output / "search_results.parquet"
    accounting = SearchAccounting()
    accounting.start_timer()

    existing = load_jsonl(jsonl_path)
    completed = completed_job_keys(existing)

    initial = plan_initial_jobs(problems, configs, settings)
    _execute_jobs(
        filter_pending(initial, completed),
        backend,
        scorer,
        settings=settings,
        jsonl_path=jsonl_path,
        accounting=accounting,
        on_record=on_record,
    )

    all_records = load_jsonl(jsonl_path)
    completed = completed_job_keys(all_records)
    adaptive = plan_adaptive_jobs(problems, configs, all_records, settings)
    _execute_jobs(
        filter_pending(adaptive, completed),
        backend,
        scorer,
        settings=settings,
        jsonl_path=jsonl_path,
        accounting=accounting,
        on_record=on_record,
    )

    accounting.stop_search_timer()
    final_records = load_jsonl(jsonl_path)
    records_to_parquet(parquet_path, final_records)
    accounting.extra["num_problems"] = len(list(problems))
    accounting.extra["num_configs"] = len(list(configs))
    accounting.extra["settings"] = asdict(settings)
    accounting.write(output / "accounting.json")
    return {
        "records": final_records,
        "jsonl_path": str(jsonl_path),
        "parquet_path": str(parquet_path),
        "accounting": accounting.to_dict(),
    }
