from __future__ import annotations

from offline_search.search.resume import append_jsonl, completed_job_keys, filter_pending, job_key, load_jsonl


def test_resume_skips_completed_job_keys(tmp_path):
    path = tmp_path / "search_results.jsonl"
    done = [
        {"problem_id": "p1", "sampling_config_id": "cfg0", "seed": 1, "response": "a"},
        {"problem_id": "p1", "sampling_config_id": "cfg0", "seed": 2, "response": "b"},
    ]
    append_jsonl(path, done)
    jobs = [
        {"problem_id": "p1", "sampling_config_id": "cfg0", "seed": 1},
        {"problem_id": "p1", "sampling_config_id": "cfg0", "seed": 2},
        {"problem_id": "p1", "sampling_config_id": "cfg0", "seed": 3},
    ]
    pending = filter_pending(jobs, completed_job_keys(load_jsonl(path)))
    assert pending == [jobs[2]]
    append_jsonl(path, [{"problem_id": "p1", "sampling_config_id": "cfg0", "seed": 3, "response": "c"}])
    rows = load_jsonl(path)
    keys = completed_job_keys(rows)
    assert len(rows) == 3
    assert len(keys) == 3
    assert job_key("p1", "cfg0", 2) in keys
