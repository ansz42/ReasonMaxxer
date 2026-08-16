from __future__ import annotations

from typing import Any

import numpy as np

def _torch():
    """Import torch only when a caller actually needs it.

    A broken Windows torch install can access-violate during import, so unit
    tests stay on the numpy path and never touch this helper.
    """
    import torch
    import torch.nn.functional as F

    return torch, F


def decision_loss_numpy(
    token_logprobs: Any,
    advantage: Any,
    token_weights: Any,
    *,
    eps: float = 1e-8,
) -> float:
    logp = np.asarray(token_logprobs, dtype=np.float64)
    weights = np.asarray(token_weights, dtype=np.float64)
    adv = np.asarray(advantage, dtype=np.float64)
    if adv.ndim == 0:
        adv = adv.reshape(1)
    while adv.ndim < logp.ndim:
        adv = np.expand_dims(adv, -1)
    if logp.shape != weights.shape:
        raise ValueError(f"logprob/weight shape mismatch: {logp.shape} vs {weights.shape}")
    numer = -float(np.sum(adv * weights * logp))
    denom = float(np.sum(weights)) + float(eps)
    return numer / denom


def _as_tensor(value: Any, torch_mod: Any):
    if isinstance(value, torch_mod.Tensor):
        return value
    return torch_mod.as_tensor(value)


def decision_loss(token_logprobs: Any, advantage: Any, token_weights: Any, *, eps: float = 1e-8):
    """Signed, length-normalized decision loss.

    L = -sum(advantage * weight * log p) / (sum(weight) + eps)
    """
    if not hasattr(token_logprobs, "sum") or isinstance(token_logprobs, (list, tuple)):
        return decision_loss_numpy(token_logprobs, advantage, token_weights, eps=eps)
    try:
        torch, _F = _torch()
    except Exception:
        return decision_loss_numpy(token_logprobs, advantage, token_weights, eps=eps)
    if not isinstance(token_logprobs, torch.Tensor):
        return decision_loss_numpy(token_logprobs, advantage, token_weights, eps=eps)

    logp = _as_tensor(token_logprobs, torch)
    weights = _as_tensor(token_weights, torch).to(dtype=logp.dtype, device=logp.device)
    adv = _as_tensor(advantage, torch).to(dtype=logp.dtype, device=logp.device)
    if adv.ndim == 0:
        adv = adv.view(1)
    while adv.ndim < logp.ndim:
        adv = adv.unsqueeze(-1)
    if logp.shape != weights.shape:
        raise ValueError(f"logprob/weight shape mismatch: {tuple(logp.shape)} vs {tuple(weights.shape)}")
    numer = -(adv * weights * logp).sum()
    denom = weights.sum() + float(eps)
    return numer / denom


def causal_token_logprobs(logits: Any, input_ids: Any):
    """log p(token_t | prefix_t) aligned to input_ids. Position 0 is 0."""
    torch, F = _torch()
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    pad = torch.zeros(gathered.size(0), 1, device=gathered.device, dtype=gathered.dtype)
    return torch.cat([pad, gathered], dim=1)


def token_entropies_from_logits(logits: Any):
    """Entropy of p(. | prefix) at each next-token position; pad position 0 with 0."""
    torch, F = _torch()
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)
    pad = torch.zeros(entropy.size(0), 1, device=entropy.device, dtype=entropy.dtype)
    return torch.cat([pad, entropy], dim=1)


def apply_response_mask(token_weights: Any, response_mask: Any):
    weights = np.asarray(token_weights, dtype=np.float64)
    mask = np.asarray(response_mask, dtype=np.float64)
    return weights * mask
