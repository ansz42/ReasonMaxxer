from __future__ import annotations

from pathlib import Path

from offline_search.training.trainer import (
    TrainSettings,
    checkpoint_dir,
    filter_informative_rows,
    resolve_max_steps,
    save_lora_checkpoint,
    should_save_checkpoint,
)
from tests.conftest import import_torch_or_skip


def test_filter_drops_zero_and_tiny_advantages():
    rows = [
        {"advantage": 0.0, "id": 1},
        {"advantage": 1.0, "id": 2},
        {"advantage": -1.0, "id": 3},
        {"advantage": 1e-12, "id": 4},
    ]
    kept = filter_informative_rows(rows)
    assert [r["id"] for r in kept] == [2, 3]


def test_filter_can_keep_zero_advantage():
    rows = [{"advantage": 0.0}, {"advantage": 1.0}]
    kept = filter_informative_rows(rows, drop_zero_advantage=False)
    assert len(kept) == 2


def test_cover_all_informative_raises_cap_to_one_pass():
    settings = TrainSettings(max_steps=2, batch_size=1, cover_all_informative=True)
    assert resolve_max_steps(5, settings) == 5


def test_cover_all_informative_does_not_shrink_cap():
    settings = TrainSettings(max_steps=20, batch_size=1, cover_all_informative=True)
    assert resolve_max_steps(5, settings) == 20


def test_cover_respects_batch_size():
    settings = TrainSettings(max_steps=1, batch_size=4, cover_all_informative=True)
    assert resolve_max_steps(10, settings) == 3


def test_trainer_skips_zero_advantage_rows():
    torch = import_torch_or_skip()
    import torch.nn as nn

    from offline_search.training.trainer import train_signed_entropy

    class TinyLM(nn.Module):
        def __init__(self, vocab: int = 32, dim: int = 16) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            return type("Out", (), {"logits": self.lm_head(self.embed(input_ids))})()

    def _row(adv: float, token: int) -> dict:
        ids = [1, token, token + 1, token + 2]
        return {
            "input_ids": ids,
            "token_weight": [0.0, 1.0, 1.0, 1.0],
            "response_mask": [0, 1, 1, 1],
            "advantage": adv,
        }

    rows = [
        _row(0.0, 3),
        _row(1.0, 5),
        _row(0.0, 7),
        _row(-1.0, 9),
        _row(0.0, 11),
    ]
    metrics = train_signed_entropy(
        TinyLM(),
        rows,
        settings=TrainSettings(
            learning_rate=1e-2,
            epochs=1,
            max_steps=8,
            batch_size=1,
            seed=0,
            logging_steps=1,
            drop_zero_advantage=True,
            cover_all_informative=True,
        ),
    )
    assert metrics["num_rows_in"] == 5
    assert metrics["num_rows"] == 2
    assert metrics["num_rows_dropped"] == 3
    assert metrics["steps"] == 8
    assert all(abs(row["advantage_mean"]) > 0.0 for row in metrics["logs"])


def test_should_save_checkpoint_every_100_steps():
    assert should_save_checkpoint(100, 100) is True
    assert should_save_checkpoint(200, 100) is True
    assert should_save_checkpoint(50, 100) is False
    assert should_save_checkpoint(0, 100) is False
    assert should_save_checkpoint(100, None) is False
    assert should_save_checkpoint(100, 0) is False


def test_save_lora_checkpoint_writes_step_dir(tmp_path):
    class Dummy:
        def save_pretrained(self, path) -> None:
            dest = Path(path)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "adapter_model.bin").write_bytes(b"lora")

    dest = save_lora_checkpoint(Dummy(), tmp_path, 100)
    assert dest == checkpoint_dir(tmp_path, 100)
    assert dest.name == "checkpoint-100"
    assert (dest / "adapter_model.bin").exists()


def test_trainer_saves_lora_every_save_steps(tmp_path):
    torch = import_torch_or_skip()
    import torch.nn as nn

    from offline_search.training.trainer import train_signed_entropy

    class TinySaveLM(nn.Module):
        def __init__(self, vocab: int = 32, dim: int = 16) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)
            self.saved: list[str] = []

        def forward(self, input_ids, attention_mask=None, **kwargs):
            del attention_mask, kwargs
            return type("Out", (), {"logits": self.lm_head(self.embed(input_ids))})()

        def save_pretrained(self, path) -> None:
            dest = Path(path)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "adapter_model.bin").write_bytes(b"lora")
            self.saved.append(str(dest))

    def _row(token: int) -> dict:
        ids = [1, token, token + 1, token + 2]
        return {
            "input_ids": ids,
            "token_weight": [0.0, 1.0, 1.0, 1.0],
            "response_mask": [0, 1, 1, 1],
            "advantage": 1.0,
        }

    model = TinySaveLM()
    metrics = train_signed_entropy(
        model,
        [_row(3), _row(5)],
        settings=TrainSettings(
            learning_rate=1e-2,
            epochs=4,
            max_steps=4,
            batch_size=1,
            seed=0,
            logging_steps=1,
            save_steps=2,
            drop_zero_advantage=True,
            cover_all_informative=True,
        ),
        output_dir=tmp_path,
    )
    expected = [str(checkpoint_dir(tmp_path, 2)), str(checkpoint_dir(tmp_path, 4)), str(tmp_path / "adapter")]
    assert metrics["checkpoints"] == expected[:2]
    assert model.saved == expected
    assert (tmp_path / "checkpoint-2" / "adapter_model.bin").exists()
    assert (tmp_path / "checkpoint-4" / "adapter_model.bin").exists()
    assert (tmp_path / "adapter" / "adapter_model.bin").exists()
