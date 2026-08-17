from __future__ import annotations

from offline_search.search.generate import ScriptedBackend, VLLMBackend, vllm_engine_kwargs


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.text = text
        self.token_ids = text.split()
        self.finish_reason = "stop"


class _FakeRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [_FakeCompletion(text)]


class FakeVLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_tokenizer(self):
        return None

    def generate(self, prompts, sampling_params, lora_request=None):
        self.calls.append(
            {
                "prompts": list(prompts),
                "n_params": len(sampling_params) if isinstance(sampling_params, list) else 1,
                "seeds": [getattr(p, "seed", None) for p in sampling_params]
                if isinstance(sampling_params, list)
                else [getattr(sampling_params, "seed", None)],
                "max_tokens": [getattr(p, "max_tokens", None) for p in sampling_params]
                if isinstance(sampling_params, list)
                else [getattr(sampling_params, "max_tokens", None)],
                "lora_request": lora_request,
            }
        )
        return [_FakeRequestOutput(f"boxed:{i}") for i, _ in enumerate(prompts)]


def test_vllm_engine_kwargs_try_eager_off_first():
    kwargs = vllm_engine_kwargs(
        model_name="unsloth/Qwen2.5-3B-Instruct",
        max_model_len=6144,
        gpu_memory_utilization=0.5,
    )
    assert kwargs["max_model_len"] == 6144
    assert kwargs["gpu_memory_utilization"] == 0.5
    assert "enforce_eager" not in kwargs


def test_vllm_engine_kwargs_opt_in_eager():
    kwargs = vllm_engine_kwargs(
        model_name="unsloth/Qwen2.5-3B-Instruct",
        max_model_len=6144,
        enforce_eager=True,
    )
    assert kwargs["enforce_eager"] is True


def test_scripted_backend_honors_per_prompt_seeds():
    backend = ScriptedBackend(
        {
            ("a", 0.5, 10): "A",
            ("b", 0.5, 20): "B",
        }
    )
    out = backend.generate(
        ["a", "b"],
        temperature=0.5,
        top_p=1.0,
        n=1,
        max_tokens=8,
        seed=0,
        seeds=[10, 20],
    )
    assert out[0][0].text == "A"
    assert out[1][0].text == "B"


def test_vllm_backend_generates_a_prompt_batch():
    llm = FakeVLLM()
    backend = VLLMBackend(llm=llm, tokenizer=None)
    out = backend.generate(
        ["p0", "p1", "p2"],
        temperature=0.7,
        top_p=0.95,
        n=1,
        max_tokens=3000,
        seed=42,
        seeds=[1, 2, 3],
    )
    assert len(out) == 3
    assert [row[0].text for row in out] == ["boxed:0", "boxed:1", "boxed:2"]
    assert len(llm.calls) == 1
    assert llm.calls[0]["prompts"] == ["p0", "p1", "p2"]
    assert llm.calls[0]["seeds"] == [1, 2, 3]
    assert llm.calls[0]["max_tokens"] == [3000, 3000, 3000]
    assert all(row[0].extra.get("rendered_prompt") == prompt for row, prompt in zip(out, ["p0", "p1", "p2"]))
