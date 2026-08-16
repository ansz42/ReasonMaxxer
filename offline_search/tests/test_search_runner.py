from __future__ import annotations

from offline_search.scoring.math_verifier import MathVerifier
from offline_search.search.generate import ScriptedBackend
from offline_search.search.sampling_configs import SamplingConfig
from offline_search.search.search_runner import Problem, SearchSettings, run_search
from offline_search.search.resume import completed_job_keys, load_jsonl


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
