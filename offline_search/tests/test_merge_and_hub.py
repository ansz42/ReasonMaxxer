from __future__ import annotations

from pathlib import Path

from offline_search.training.merge import (
    adapter_looks_complete,
    parse_checkpoint_step,
    resolve_hub_repo_id,
    resolve_latest_adapter,
)


def _write_adapter(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"x")


def test_parse_checkpoint_step():
    assert parse_checkpoint_step(Path("checkpoint-250")) == 250
    assert parse_checkpoint_step(Path("adapter")) is None
    assert parse_checkpoint_step(Path("checkpoint-foo")) is None


def test_resolve_latest_prefers_final_adapter(tmp_path: Path):
    train = tmp_path / "train"
    _write_adapter(train / "checkpoint-50")
    _write_adapter(train / "checkpoint-250")
    _write_adapter(train / "adapter")
    assert resolve_latest_adapter(train) == train / "adapter"


def test_resolve_latest_uses_highest_checkpoint(tmp_path: Path):
    train = tmp_path / "train"
    _write_adapter(train / "checkpoint-50")
    _write_adapter(train / "checkpoint-200")
    _write_adapter(train / "checkpoint-150")
    assert resolve_latest_adapter(train) == train / "checkpoint-200"
    assert adapter_looks_complete(train / "checkpoint-200")


def test_resolve_latest_missing(tmp_path: Path):
    try:
        resolve_latest_adapter(tmp_path / "missing")
    except FileNotFoundError as exc:
        assert "adapter" in str(exc).lower()
    else:
        raise AssertionError("expected FileNotFoundError")


def test_resolve_hub_repo_id_passthrough_and_name():
    assert resolve_hub_repo_id(name="math-test-maxx", repo_id="alice/math-test-maxx") == "alice/math-test-maxx"
    assert resolve_hub_repo_id(name="bob/math-test-maxx") == "bob/math-test-maxx"
    assert resolve_hub_repo_id(name="math-test-maxx", username="carol") == "carol/math-test-maxx"
