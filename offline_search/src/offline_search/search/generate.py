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
        seeds: Sequence[int] | None = None,
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
        seeds: Sequence[int] | None = None,
    ) -> list[list[GenerationResult]]:
        del top_p, max_tokens, top_k, repetition_penalty
        prompt_seeds = [int(s) for s in seeds] if seeds is not None else [int(seed)] * len(prompts)
        if len(prompt_seeds) != len(prompts):
            raise ValueError("seeds must match prompts")
        self.calls.append(
            {
                "prompts": list(prompts),
                "temperature": float(temperature),
                "n": int(n),
                "seed": int(seed),
                "seeds": prompt_seeds,
            }
        )
        out: list[list[GenerationResult]] = []
        for prompt, prompt_seed in zip(prompts, prompt_seeds):
            rows: list[GenerationResult] = []
            for i in range(int(n)):
                key = (prompt, float(temperature), int(prompt_seed) + i)
                text = self.script.get(key)
                if text is None:
                    text = self.script.get((prompt, float(temperature), int(prompt_seed)), f"scripted:{prompt_seed}:{i}")
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
        seeds: Sequence[int] | None = None,
    ) -> list[list[GenerationResult]]:
        import torch

        from offline_search.prompting import render_generation_prompt

        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError("seeds must match prompts")
        if seeds is not None and len(set(int(s) for s in seeds)) > 1:
            results: list[list[GenerationResult]] = []
            for prompt, prompt_seed in zip(prompts, seeds):
                results.extend(
                    self.generate(
                        [prompt],
                        temperature=temperature,
                        top_p=top_p,
                        n=n,
                        max_tokens=max_tokens,
                        seed=int(prompt_seed),
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )
                )
            return results

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
        seeds: Sequence[int] | None = None,
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
            seeds=seeds,
        )


def _prompt_seeds(prompts: Sequence[str], seed: int, seeds: Sequence[int] | None) -> list[int]:
    if seeds is None:
        return [int(seed) + i for i in range(len(prompts))]
    values = [int(s) for s in seeds]
    if len(values) != len(prompts):
        raise ValueError("seeds must match prompts")
    return values


def vllm_engine_kwargs(
    *,
    model_name: str,
    max_model_len: int,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.5,
    enforce_eager: bool = False,
    adapter_path: str | None = None,
    max_lora_rank: int = 64,
) -> dict[str, Any]:
    """Build LLM() kwargs. Eager mode stays off unless the caller opts in."""
    kwargs: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": True,
        "max_model_len": int(max_model_len),
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": float(gpu_memory_utilization),
    }
    if enforce_eager:
        kwargs["enforce_eager"] = True
    if adapter_path:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = int(max_lora_rank)
    return kwargs


class VLLMBackend:
    """Batched generation via vLLM. Training/entropy still use the Unsloth stack."""

    def __init__(
        self,
        llm: Any = None,
        tokenizer: Any = None,
        *,
        model_name: str | None = None,
        max_model_len: int = 8192,
        enable_thinking: bool = False,
        adapter_path: str | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        enforce_eager: bool = False,
        max_lora_rank: int = 64,
    ) -> None:
        self.enable_thinking = enable_thinking
        self.adapter_path = str(adapter_path) if adapter_path else None
        if llm is None:
            if not model_name:
                raise ValueError("VLLMBackend needs llm= or model_name=")
            from vllm import LLM

            llm = LLM(
                **vllm_engine_kwargs(
                    model_name=model_name,
                    max_model_len=max_model_len,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    enforce_eager=enforce_eager,
                    adapter_path=self.adapter_path,
                    max_lora_rank=max_lora_rank,
                )
            )
        self.llm = llm
        self.tokenizer = tokenizer if tokenizer is not None else self.llm.get_tokenizer()

    def _lora_request(self) -> Any:
        if not self.adapter_path:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest("offline_search", 1, self.adapter_path)

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
        seeds: Sequence[int] | None = None,
    ) -> list[list[GenerationResult]]:
        from offline_search.prompting import render_generation_prompt

        try:
            from vllm import SamplingParams
        except ImportError:  # unit tests inject a fake LLM
            from types import SimpleNamespace as SamplingParams

        rendered = [
            render_generation_prompt(self.tokenizer, prompt, enable_thinking=self.enable_thinking) for prompt in prompts
        ]
        prompt_seeds = _prompt_seeds(prompts, seed, seeds)
        top_k_value = int(top_k) if top_k is not None else -1
        do_sample = float(temperature) > 0
        params: list[Any] = []
        for prompt_seed in prompt_seeds:
            kwargs: dict[str, Any] = {
                "n": int(n),
                "temperature": max(float(temperature), 1e-5) if do_sample else 0.0,
                "top_p": float(top_p) if do_sample else 1.0,
                "max_tokens": int(max_tokens),
                "seed": int(prompt_seed),
                "repetition_penalty": float(repetition_penalty),
            }
            if top_k_value > 0:
                kwargs["top_k"] = top_k_value
            params.append(SamplingParams(**kwargs))

        lora_request = self._lora_request()
        if lora_request is not None:
            outputs = self.llm.generate(rendered, params, lora_request=lora_request)
        else:
            outputs = self.llm.generate(rendered, params)

        results: list[list[GenerationResult]] = []
        for rendered_prompt, request in zip(rendered, outputs):
            rows: list[GenerationResult] = []
            for completion in request.outputs:
                token_ids = getattr(completion, "token_ids", None) or []
                rows.append(
                    GenerationResult(
                        text=str(completion.text),
                        num_tokens=max(1, len(token_ids)),
                        finish_reason=getattr(completion, "finish_reason", None),
                        extra={"rendered_prompt": rendered_prompt},
                    )
                )
            results.append(rows)
        return results
