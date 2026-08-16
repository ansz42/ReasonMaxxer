from __future__ import annotations

from offline_search.data.build_training_dataset import CharTokenizer, build_training_rows, write_training_dataset
from offline_search.data.select_trajectories import SelectionCaps
from offline_search.eval.generate_eval import evaluate_backend
from offline_search.scoring.math_verifier import MathVerifier
from offline_search.search.generate import ScriptedBackend
from offline_search.search.sampling_configs import SamplingConfig
from offline_search.search.search_runner import Problem, SearchSettings, run_search
from offline_search.utils.seeds import stable_seed
from tests.conftest import import_torch_or_skip


def _run_search_and_dataset(tmp_path):
    problem = Problem("add", "What is 17+28?", "45")
    configs = [
        SamplingConfig("explore", temperature=1.1, top_p=1.0),
        SamplingConfig("exploit", temperature=0.4, top_p=0.9),
    ]
    settings = SearchSettings(
        initial_samples_per_config=2,
        total_samples_per_problem=6,
        exploration_fraction=0.3,
        seed=3,
        max_tokens=32,
    )
    script = {}
    for sample_i in range(12):
        s_ex = stable_seed(problem.problem_id, "explore", sample_i, base=settings.seed)
        s_ex2 = stable_seed(problem.problem_id, "exploit", sample_i, base=settings.seed)
        script[(problem.prompt, 1.1, s_ex)] = "Add the numbers. \\boxed{45}"
        script[(problem.prompt, 0.4, s_ex2)] = "I guess \\boxed{40}"
    backend = ScriptedBackend(script)
    search = run_search([problem], configs, backend, MathVerifier(), tmp_path / "search", settings)
    rows = build_training_rows(
        search["records"],
        tokenizer=CharTokenizer(),
        caps=SelectionCaps(4, 2, 2, 1),
        objective="graded_signed",
    )
    paths = write_training_dataset(rows, tmp_path / "data")
    return problem, search, rows, paths


def test_synthetic_search_dataset_eval(tmp_path):
    problem, search, rows, paths = _run_search_and_dataset(tmp_path)
    assert search["accounting"]["generated_trajectories"] == 6
    assert any(r["is_correct"] for r in search["records"])
    assert rows
    assert (tmp_path / "data" / "train_entropy.parquet").exists()
    assert paths["parquet"].endswith("train_entropy.parquet")

    eval_backend = ScriptedBackend(
        {(problem.prompt, 0.6, stable_seed("eval", problem.problem_id, 0, base=42)): "\\boxed{45}"}
    )
    report = evaluate_backend(
        [problem],
        eval_backend,
        MathVerifier(),
        n_samples=1,
        temperature=0.6,
        top_p=0.95,
        max_tokens=16,
        ks=[1],
        output_path=tmp_path / "eval.json",
    )
    assert report["macro"]["pass@1"] == 1.0


def test_synthetic_tiny_train(tmp_path):
    torch = import_torch_or_skip()
    import torch.nn as nn

    from offline_search.training.trainer import TrainSettings, train_signed_entropy

    class TinyLM(nn.Module):
        def __init__(self, vocab: int = 128, dim: int = 32) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            return type("Out", (), {"logits": self.lm_head(self.embed(input_ids))})()

    _, _, rows, _ = _run_search_and_dataset(tmp_path)
    model = TinyLM()
    metrics = train_signed_entropy(
        model,
        rows,
        settings=TrainSettings(learning_rate=1e-2, epochs=1, max_steps=2, batch_size=1, seed=0),
        output_dir=tmp_path / "train",
    )
    assert metrics["steps"] >= 1
    assert (tmp_path / "train" / "train_metrics.json").exists()
