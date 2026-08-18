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


def _prepare_numpy(token_logprobs: Any, advantage: Any, token_weights: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logp = np.asarray(token_logprobs, dtype=np.float64)
    weights = np.asarray(token_weights, dtype=np.float64)
    adv = np.asarray(advantage, dtype=np.float64)
    if logp.shape != weights.shape:
        raise ValueError(f"logprob/weight shape mismatch: {logp.shape} vs {weights.shape}")
    if logp.ndim == 1:
        logp = logp.reshape(1, -1)
        weights = weights.reshape(1, -1)
    if adv.ndim == 0:
        adv = np.full((logp.shape[0],), float(adv), dtype=np.float64)
    else:
        adv = adv.reshape(-1)
    if adv.shape[0] == 1 and logp.shape[0] > 1:
        adv = np.broadcast_to(adv, (logp.shape[0],)).copy()
    if adv.shape[0] != logp.shape[0]:
        raise ValueError(f"advantage/batch mismatch: {adv.shape[0]} vs {logp.shape[0]}")
    return logp, adv, weights


def _sequence_losses_numpy(
    logp: np.ndarray,
    adv: np.ndarray,
    weights: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    token = adv[:, None] * weights * logp
    numer = -token.sum(axis=-1)
    mass = weights.sum(axis=-1)
    return numer / (mass + float(eps)), mass


def decision_loss_numpy(
    token_logprobs: Any,
    advantage: Any,
    token_weights: Any,
    *,
    eps: float = 1e-8,
) -> float:
    logp, adv, weights = _prepare_numpy(token_logprobs, advantage, token_weights)
    per_seq, mass = _sequence_losses_numpy(logp, adv, weights, eps=eps)
    valid = mass > 0
    if not np.any(valid):
        return 0.0
    return float(per_seq[valid].mean())


def apply_neg_prob_floor(
    token_logprobs: Any,
    advantage: Any,
    token_weights: Any,
    floor: float,
):
    """Zero negative-advantage weights once p(token) is already <= floor."""
    if float(floor) <= 0.0:
        return token_weights
    if hasattr(token_logprobs, "exp") and not isinstance(token_logprobs, (list, tuple, np.ndarray)):
        try:
            torch, _F = _torch()
        except Exception:
            torch = None
        if torch is not None and isinstance(token_logprobs, torch.Tensor):
            logp = token_logprobs
            weights = token_weights.to(dtype=logp.dtype, device=logp.device)
            adv = advantage.to(dtype=logp.dtype, device=logp.device)
            if adv.ndim == 0:
                adv = adv.view(1)
            while adv.ndim < logp.ndim:
                adv = adv.unsqueeze(-1)
            dead = logp.exp() <= float(floor)
            return torch.where((adv < 0) & dead, torch.zeros_like(weights), weights)

    logp, adv, weights = _prepare_numpy(token_logprobs, advantage, token_weights)
    dead = np.exp(logp) <= float(floor)
    neg = adv[:, None] < 0
    out = np.where(neg & dead, 0.0, weights)
    if isinstance(token_weights, list):
        if np.asarray(token_weights).ndim == 1:
            return out[0].tolist()
        return out.tolist()
    if np.asarray(token_logprobs).ndim == 1:
        return out[0]
    return out


def loss_diagnostics_numpy(
    token_logprobs: Any,
    advantage: Any,
    token_weights: Any,
    *,
    eps: float = 1e-8,
) -> dict[str, float]:
    logp, adv, weights = _prepare_numpy(token_logprobs, advantage, token_weights)
    pos = adv > 0
    neg = adv < 0

    def _weighted_logp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        w = weights[mask]
        mass = float(w.sum())
        if mass <= 0.0:
            return 0.0
        return float((w * logp[mask]).sum() / (mass + float(eps)))

    mean_logp_pos = _weighted_logp(pos)
    mean_logp_neg = _weighted_logp(neg)
    seq_mass = weights.sum(axis=-1)
    mass_sum = float(seq_mass.sum())
    if mass_sum > 0.0:
        adv_w = float((adv * seq_mass).sum() / mass_sum)
    else:
        adv_w = 0.0
    return {
        "mean_logp_pos": mean_logp_pos,
        "mean_logp_neg": mean_logp_neg,
        "mean_ce_pos": (-mean_logp_pos if np.any(pos) else 0.0),
        "mean_ce_neg": (-mean_logp_neg if np.any(neg) else 0.0),
        "advantage_mean": float(adv.mean()) if adv.size else 0.0,
        "advantage_weight_mean": adv_w,
    }


def loss_diagnostics(token_logprobs: Any, advantage: Any, token_weights: Any, *, eps: float = 1e-8) -> dict[str, float]:
    if hasattr(token_logprobs, "detach"):
        return loss_diagnostics_numpy(
            token_logprobs.detach().cpu().numpy(),
            advantage.detach().cpu().numpy() if hasattr(advantage, "detach") else advantage,
            token_weights.detach().cpu().numpy() if hasattr(token_weights, "detach") else token_weights,
            eps=eps,
        )
    return loss_diagnostics_numpy(token_logprobs, advantage, token_weights, eps=eps)


def _as_tensor(value: Any, torch_mod: Any):
    if isinstance(value, torch_mod.Tensor):
        return value
    return torch_mod.as_tensor(value)


def decision_loss(token_logprobs: Any, advantage: Any, token_weights: Any, *, eps: float = 1e-8):
    """Signed, per-sequence then batch-mean decision loss.

    For each sequence i:
        L_i = -sum_t(A_i * w_it * log p_it) / (sum_t w_it + eps)
    Then L = mean of L_i over sequences with nonzero weight.
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
    token = adv * weights * logp
    numer = -token.sum(dim=-1)
    mass = weights.sum(dim=-1)
    per_seq = numer / (mass + float(eps))
    valid = mass > 0
    if bool(valid.any()):
        return per_seq[valid].mean()
    return per_seq.new_zeros(())


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
