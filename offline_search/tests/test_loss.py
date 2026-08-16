from __future__ import annotations

import math

from offline_search.training.loss import apply_response_mask, decision_loss, decision_loss_numpy
from tests.conftest import import_torch_or_skip


def test_zero_weight_token_does_not_change_loss():
    logp = [math.log(0.2), math.log(0.4)]
    w = [1.0, 0.0]
    loss_a = decision_loss_numpy(logp, 1.0, w)
    loss_b = decision_loss_numpy(logp + [math.log(0.01)], 1.0, w + [0.0])
    assert abs(loss_a - loss_b) < 1e-12


def test_prompt_tokens_do_not_contribute():
    logp = [[math.log(0.3), math.log(0.5)]]
    weights = apply_response_mask([[1.0, 1.0]], [[0.0, 1.0]])
    loss = decision_loss_numpy(logp, 2.0, weights)
    expected = decision_loss_numpy([[0.0, math.log(0.5)]], 2.0, [[0.0, 1.0]])
    assert abs(loss - expected) < 1e-12


def test_positive_advantage_has_negative_logprob_gradient():
    plus = decision_loss_numpy([math.log(0.25) + 1e-4], 1.0, [1.0])
    minus = decision_loss_numpy([math.log(0.25) - 1e-4], 1.0, [1.0])
    assert plus < minus


def test_negative_advantage_has_positive_logprob_gradient():
    plus = decision_loss_numpy([math.log(0.25) + 1e-4], -1.0, [1.0])
    minus = decision_loss_numpy([math.log(0.25) - 1e-4], -1.0, [1.0])
    assert plus > minus


def test_length_normalization_ignores_zero_weight_padding():
    logp = [math.log(0.25)]
    w = [1.0]
    loss_short = decision_loss_numpy(logp, -1.5, w)
    loss_long = decision_loss_numpy(logp + [math.log(0.01), math.log(0.02)], -1.5, w + [0.0, 0.0])
    assert abs(loss_short - loss_long) < 1e-12


def test_torch_and_numpy_match():
    torch = import_torch_or_skip()
    logp = torch.tensor([[-1.0, -2.0]], dtype=torch.float64)
    w = torch.tensor([[1.0, 0.25]], dtype=torch.float64)
    adv = torch.tensor([0.5], dtype=torch.float64)
    torch_loss = float(decision_loss(logp, adv, w).detach())
    numpy_loss = decision_loss_numpy(logp.numpy(), adv.numpy(), w.numpy())
    assert abs(torch_loss - numpy_loss) < 1e-9


def _one_step(advantage: float) -> tuple[float, float]:
    torch = import_torch_or_skip()
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyLM(nn.Module):
        def __init__(self, vocab: int = 8, dim: int = 16) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)
            nn.init.normal_(self.embed.weight, std=0.3)
            nn.init.normal_(self.lm_head.weight, std=0.3)

        def forward(self, input_ids):
            return type("Out", (), {"logits": self.lm_head(self.embed(input_ids))})()

    torch.manual_seed(0)
    model = TinyLM()
    ids = torch.tensor([[0, 3]], dtype=torch.long)

    def target_logprob() -> torch.Tensor:
        logits = model(ids).logits
        logp = F.log_softmax(logits[0, 0], dim=-1)
        return logp[ids[0, 1]]

    before = float(target_logprob().detach())
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    logits = model(ids).logits
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    token_logprobs = torch.zeros_like(ids, dtype=logp.dtype)
    token_logprobs[:, 1:] = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    weights = torch.tensor([[0.0, 1.0]], dtype=logp.dtype)
    adv = torch.tensor([advantage], dtype=logp.dtype)
    loss = decision_loss(token_logprobs, adv, weights)
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = float(target_logprob().detach())
    return before, after


def test_positive_advantage_increases_token_probability():
    before, after = _one_step(1.0)
    assert after > before


def test_negative_advantage_decreases_token_probability():
    before, after = _one_step(-1.0)
    assert after < before
