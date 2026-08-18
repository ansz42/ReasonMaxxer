from __future__ import annotations

from offline_search.scoring.math_verifier import MathVerifier
from offline_search.search.clipping import clip_retry_seed
from offline_search.search.generate import ScriptedBackend
from offline_search.search.sampling_configs import SamplingConfig
from offline_search.search.search_runner import Problem, SearchSettings, run_search
from offline_search.search.resume import completed_job_keys, load_jsonl
from offline_search.utils.seeds import stable_seed


def _configs() -> list[SamplingConfig]:
    return [
        SamplingConfig("hot", temperature=1.2, top_p=1.0),
        SamplingConfig("cold", temperature=0.3, top_p=0.9),
    ]


def test_search_is_resumable_without_duplicate_keys(tmp_path):
    problem = Problem("add", "What is 2+2?", "4")
    prompt = problem.prompt
    script = {
        (prompt, 1.2, 1): "\\boxed{4}",
        (prompt, 0.3, 1): "\\boxed{5}",
    }
    backend = ScriptedBackend(script)
    settings = SearchSettings(
        initial_samples_per_config=1,
        total_samples_per_problem=2,
        exploration_fraction=0.5,
        seed=7,
        max_tokens=16,
    )
    first = run_search([problem], _configs(), backend, MathVerifier(), tmp_path / "run", settings)
    n_first = len(first["records"])
    second = run_search([problem], _configs(), backend, MathVerifier(), tmp_path / "run", settings)
    rows = load_jsonl(tmp_path / "run" / "search_results.jsonl")
    assert len(second["records"]) == n_first
    assert len(rows) == n_first
    assert len(completed_job_keys(rows)) == n_first
    assert (tmp_path / "run" / "search_results.parquet").exists()


def test_adaptive_stage_prefers_better_config(tmp_path):
    problem = Problem("add", "What is 2+2?", "4")
    configs = _configs()
    settings = SearchSettings(
        initial_samples_per_config=2,
        total_samples_per_problem=8,
        exploration_fraction=0.25,
        allocation_temperature=0.3,
        seed=0,
        max_tokens=16,
    )
    # Build a backend that is correct only for the hot arm.
    from offline_search.utils.seeds import stable_seed

    script = {}
    for sample_i in range(16):
        hot_seed = stable_seed(problem.problem_id, "hot", sample_i, base=settings.seed)
        cold_seed = stable_seed(problem.problem_id, "cold", sample_i, base=settings.seed)
        script[(problem.prompt, 1.2, hot_seed)] = "\\boxed{4}"
        script[(problem.prompt, 0.3, cold_seed)] = "\\boxed{9}"
    backend = ScriptedBackend(script)
    result = run_search([problem], configs, backend, MathVerifier(), tmp_path / "adapt", settings)
    counts = {"hot": 0, "cold": 0}
    for rec in result["records"]:
        counts[rec["sampling_config_id"]] += 1
    assert sum(counts.values()) == 8
    assert counts["hot"] > counts["cold"]


def test_search_batches_jobs_that_share_sampling_params(tmp_path):
    problems = [Problem(f"p{i}", f"prompt-{i}", "1") for i in range(6)]
    configs = [SamplingConfig("hot", temperature=0.8, top_p=1.0)]
    settings = SearchSettings(
        initial_samples_per_config=1,
        total_samples_per_problem=1,
        generation_batch_size=4,
        seed=0,
        max_tokens=8,
    )
    backend = ScriptedBackend()
    result = run_search(problems, configs, backend, MathVerifier(), tmp_path / "batch", settings)
    assert len(result["records"]) == 6
    sizes = [len(call["prompts"]) for call in backend.calls]
    assert sizes == [4, 2]
    assert all(call["n"] == 1 for call in backend.calls)
    assert all(call.get("seeds") and len(call["seeds"]) == len(call["prompts"]) for call in backend.calls)


def test_search_retries_clipped_once_and_keeps_replacement(tmp_path):
    problem = Problem("add", "What is 2+2?", "4")
    configs = [SamplingConfig("hot", temperature=1.2, top_p=1.0)]
    settings = SearchSettings(
        initial_samples_per_config=1,
        total_samples_per_problem=1,
        seed=7,
        max_tokens=16,
        retry_clipped=True,
    )
    seed = stable_seed(problem.problem_id, "hot", 0, base=settings.seed)
    retry = clip_retry_seed(seed)
    backend = ScriptedBackend(
        {
            (problem.prompt, 1.2, seed): "clipped first pass",
            (problem.prompt, 1.2, retry): "\\boxed{4}",
        },
        finish_reasons={
            (problem.prompt, 1.2, seed): "length",
            (problem.prompt, 1.2, retry): "stop",
        },
    )
    result = run_search([problem], configs, backend, MathVerifier(), tmp_path / "clip-ok", settings)
    kept = [r for r in result["records"] if not r.get("discarded")]
    assert len(kept) == 1
    assert kept[0]["response"] == "\\boxed{4}"
    assert kept[0]["clip_retried"] is True
    assert kept[0]["is_correct"] is True
    assert kept[0]["seed"] == seed
    assert len(backend.calls) == 2


class _AlwaysClippedBackend(ScriptedBackend):
    def generate(self, prompts, **kwargs):
        rows = super().generate(prompts, **kwargs)
        for group in rows:
            for result in group:
                result.finish_reason = "length"
        return rows


def test_search_discards_if_retry_still_clipped(tmp_path):
    problem = Problem("add", "What is 2+2?", "4")
    configs = [SamplingConfig("hot", temperature=1.2, top_p=1.0)]
    settings = SearchSettings(
        initial_samples_per_config=1,
        total_samples_per_problem=1,
        seed=7,
        max_tokens=16,
        retry_clipped=True,
    )
    backend = _AlwaysClippedBackend()
    result = run_search([problem], configs, backend, MathVerifier(), tmp_path / "clip-fail", settings)
    kept = [r for r in result["records"] if not r.get("discarded")]
    discarded = [r for r in result["records"] if r.get("discarded")]
    assert kept == []
    assert len(discarded) >= 1
    assert all(r["clipped"] and r["clip_retried"] and r["discarded"] for r in discarded)
    assert result["accounting"].get("clipped_discarded") == len(discarded)


def test_adaptive_backfills_discarded_clipped_slots(tmp_path):
    problem = Problem("add", "What is 2+2?", "4")
    configs = [SamplingConfig("hot", temperature=1.2, top_p=1.0)]
    settings = SearchSettings(
        initial_samples_per_config=1,
        total_samples_per_problem=2,
        exploration_fraction=0.0,
        seed=7,
        max_tokens=16,
        retry_clipped=True,
    )
    first = stable_seed(problem.problem_id, "hot", 0, base=settings.seed)
    first_retry = clip_retry_seed(first)
    script = {
        (problem.prompt, 1.2, first): "clipped first",
        (problem.prompt, 1.2, first_retry): "clipped again",
    }
    finish_reasons = {
        (problem.prompt, 1.2, first): "length",
        (problem.prompt, 1.2, first_retry): "length",
    }
    for sample_i in range(1, 6):
        seed = stable_seed(problem.problem_id, "hot", sample_i, base=settings.seed)
        script[(problem.prompt, 1.2, seed)] = "\\boxed{4}"
        finish_reasons[(problem.prompt, 1.2, seed)] = "stop"
    backend = ScriptedBackend(script, finish_reasons=finish_reasons)
    result = run_search([problem], configs, backend, MathVerifier(), tmp_path / "clip-fill", settings)
    kept = [r for r in result["records"] if not r.get("discarded")]
    assert len(kept) == 2
    assert all(r["is_correct"] for r in kept)
