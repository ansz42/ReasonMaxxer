from __future__ import annotations

from typing import Any, Callable, Sequence

from offline_search.data.entropy_weights import entropy_weights

EntropyFn = Callable[[Sequence[int]], list[float]]


def response_mask_for_ids(input_ids: Sequence[int], prompt_length: int) -> list[int]:
    mask = [0] * len(input_ids)
    start = max(1, int(prompt_length))
    for i in range(start, len(input_ids)):
        mask[i] = 1
    return mask


def labels_for_ids(input_ids: Sequence[int], prompt_length: int) -> list[int]:
    labels = list(input_ids)
    cutoff = max(1, int(prompt_length))
    for i in range(cutoff):
        labels[i] = -100
    return labels


def attach_entropy_and_weights(
    input_ids: Sequence[int],
    prompt_length: int,
    entropy_fn: EntropyFn,
    *,
    threshold: float,
    scale: float,
    mode: str = "hard",
) -> dict[str, Any]:
    ids = [int(x) for x in input_ids]
    entropies = [float(x) for x in entropy_fn(ids)]
    if len(entropies) != len(ids):
        raise ValueError(f"entropy_fn returned {len(entropies)} values for {len(ids)} tokens")
    mask = response_mask_for_ids(ids, prompt_length)
    raw_weights = entropy_weights(entropies, threshold=threshold, scale=scale, mode=mode)
    token_weight = [w * m for w, m in zip(raw_weights, mask)]
    return {
        "input_ids": ids,
        "labels": labels_for_ids(ids, prompt_length),
        "response_mask": mask,
        "token_entropy": entropies,
        "token_weight": token_weight,
        "prompt_length": int(prompt_length),
    }


def uniform_entropy_fn(ids: Sequence[int], value: float = 1.0) -> list[float]:
    return [0.0] + [float(value)] * max(0, len(ids) - 1)


def model_entropy_fn(model: Any, *, device: str | None = None) -> EntropyFn:
    def _fn(ids: Sequence[int]) -> list[float]:
        import torch

        from offline_search.training.loss import token_entropies_from_logits

        tensor = torch.tensor([list(ids)], dtype=torch.long)
        if device:
            tensor = tensor.to(device)
            model_ref = model.to(device)
        else:
            model_ref = model
        model_ref.eval()
        with torch.no_grad():
            outputs = model_ref(tensor)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            entropy = token_entropies_from_logits(logits)[0]
        return [float(x) for x in entropy.detach().cpu().tolist()]

    return _fn
