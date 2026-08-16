from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass
class GenerationResult:
    text: str
    num_tokens: int
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class GenerationBackend(Protocol):
    def generate(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        n: int,
        max_tokens: int,
        seed: int,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[list[GenerationResult]]:
        ...


class ScriptedBackend:
    """Deterministic backend for tests. Maps (prompt, temperature, seed) -> text."""

    def __init__(self, script: dict[tuple[str, float, int], str] | None = None) -> None:
        self.script = script or {}
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        n: int,
        max_tokens: int,
        seed: int,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[list[GenerationResult]]:
        del top_p, max_tokens, top_k, repetition_penalty
        self.calls.append(
            {
                "prompts": list(prompts),
                "temperature": float(temperature),
                "n": int(n),
                "seed": int(seed),
            }
        )
        out: list[list[GenerationResult]] = []
        for prompt in prompts:
            rows: list[GenerationResult] = []
            for i in range(int(n)):
                key = (prompt, float(temperature), int(seed) + i)
                text = self.script.get(key)
                if text is None:
                    text = self.script.get((prompt, float(temperature), int(seed)), f"scripted:{seed}:{i}")
                token_count = max(1, len(text.split()))
                rows.append(GenerationResult(text=text, num_tokens=token_count))
            out.append(rows)
        return out


class TransformersBackend:
    def __init__(self, model: Any, tokenizer: Any, *, enable_thinking: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def generate(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        n: int,
        max_tokens: int,
        seed: int,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[list[GenerationResult]]:
        import torch

        from offline_search.prompting import render_generation_prompt

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device = getattr(self.model, "device", None)
        if device is None:
            device = next(self.model.parameters()).device
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "max_length", None) not in (None, 0):
            try:
                gen_cfg.max_length = None
            except Exception:
                pass

        # Same prefix string later reused by tokenize_pair / entropy.
        rendered = [
            render_generation_prompt(self.tokenizer, prompt, enable_thinking=self.enable_thinking) for prompt in prompts
        ]
        results: list[list[GenerationResult]] = [[] for _ in prompts]
        do_sample = float(temperature) > 0
        for sample_i in range(int(n)):
            # Chat templates already emit special tokens; do not add a second BOS.
            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": int(max_tokens),
                "do_sample": do_sample,
                "temperature": max(float(temperature), 1e-5) if do_sample else None,
                "top_p": float(top_p) if do_sample else None,
                "repetition_penalty": float(repetition_penalty),
            }
            if top_k is not None and do_sample:
                gen_kwargs["top_k"] = int(top_k)
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            with torch.no_grad():
                outputs = self.model.generate(**encoded, **gen_kwargs)
            prompt_lens = encoded["attention_mask"].sum(dim=1).tolist()
            for row_i, (seq, prompt_len) in enumerate(zip(outputs, prompt_lens)):
                gen_ids = seq[int(prompt_len) :]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                results[row_i].append(
                    GenerationResult(
                        text=text,
                        num_tokens=int(len(gen_ids)),
                        extra={"rendered_prompt": rendered[row_i]},
                    )
                )
        return results


class UnslothBackend:
    def __init__(self, model: Any, tokenizer: Any, *, enable_thinking: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    def generate(
        self,
        prompts: Sequence[str],
        *,
        temperature: float,
        top_p: float,
        n: int,
        max_tokens: int,
        seed: int,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> list[list[GenerationResult]]:
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(self.model)
        except Exception:
            pass
        inner = TransformersBackend(self.model, self.tokenizer, enable_thinking=self.enable_thinking)
        return inner.generate(
            prompts,
            temperature=temperature,
            top_p=top_p,
            n=n,
            max_tokens=max_tokens,
            seed=seed,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
