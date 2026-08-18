from __future__ import annotations

from collections import UserDict

from offline_search.data.build_training_dataset import build_training_rows, tokenize_pair
from offline_search.prompting import (
    encode_generation_prefix,
    encode_ids,
    encode_training_sequence,
    render_generation_prompt,
)
from offline_search.search.generate import GenerationResult
from offline_search.search.sampling_configs import SamplingConfig
from offline_search.search.search_runner import Problem, SearchSettings, run_search
from offline_search.scoring.math_verifier import MathVerifier


class FakeChatTokenizer:
    """Mimics a Qwen-style chat tokenizer without downloading weights."""

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
        **kwargs,
    ):
        del tokenize, kwargs
        user = messages[0]["content"]
        text = f"<|im_start|>user\n{user}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        if enable_thinking:
            text += "<think>\n"
        return text

    def __call__(self, text, add_special_tokens=False, **kwargs):
        del kwargs
        ids = [1] if add_special_tokens else []
        ids.extend((ord(ch) % 80) + 5 for ch in text)
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]


class BatchEncodingLike(UserDict):
    """Mimics transformers.BatchEncoding: UserDict, not dict; iter yields keys."""


class HfBatchEncodingTokenizer(FakeChatTokenizer):
    """Same chat tokenizer, but __call__ returns a BatchEncoding-shaped mapping."""

    def __call__(self, text, add_special_tokens=False, **kwargs):
        ids = super().__call__(text, add_special_tokens=add_special_tokens, **kwargs)["input_ids"]
        return BatchEncodingLike({"input_ids": ids, "attention_mask": [1] * len(ids)})


def test_encode_ids_unwraps_batch_encoding_mapping():
    tok = HfBatchEncodingTokenizer()
    ids = encode_ids(tok, "hello", add_special_tokens=False)
    expected = list(tok("hello", add_special_tokens=False)["input_ids"])
    assert ids == expected
    assert ids and all(isinstance(x, int) for x in ids)


def test_encode_generation_prefix_accepts_batch_encoding_tokenizer():
    tok = HfBatchEncodingTokenizer()
    rendered, prefix_ids = encode_generation_prefix(tok, "Solve this")
    expected = list(tok(rendered, add_special_tokens=False)["input_ids"])
    assert prefix_ids == expected


def test_generation_prefix_is_chat_template_not_raw_prompt():
    tok = FakeChatTokenizer()
    prompt = "Solve this math problem"
    rendered = render_generation_prompt(tok, prompt, enable_thinking=False)
    assert rendered.startswith("<|im_start|>user\n")
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "Solve this math problem" in rendered

    _, prefix_ids = encode_generation_prefix(tok, prompt)
    raw_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert prefix_ids != raw_ids
    assert prefix_ids == tok(rendered, add_special_tokens=False)["input_ids"]


def test_tokenize_pair_appends_response_to_generation_prefix():
    tok = FakeChatTokenizer()
    prompt = "Solve this..."
    response = "\\boxed{42}"
    ids, prompt_len = tokenize_pair(tok, prompt, response)

    rendered, prefix_ids = encode_generation_prefix(tok, prompt)
    response_ids = tok(response, add_special_tokens=False)["input_ids"]
    assert ids == prefix_ids + response_ids
    assert prompt_len == len(prefix_ids)
    assert ids[:prompt_len] == prefix_ids

    concat_ids = tok(prompt + response, add_special_tokens=False)["input_ids"]
    assert ids != concat_ids
    assert prompt_len != len(tok(prompt, add_special_tokens=False)["input_ids"])
    assert rendered.startswith("<|im_start|>user\n")


def test_tokenize_pair_reuses_stored_rendered_prefix():
    tok = FakeChatTokenizer()
    ids, prompt_len = tokenize_pair(
        tok,
        "ignored user text",
        "ANS",
        rendered_prefix="CUSTOM_PREFIX",
    )
    prefix_ids = tok("CUSTOM_PREFIX", add_special_tokens=False)["input_ids"]
    assert ids[:prompt_len] == prefix_ids
    assert ids[prompt_len:] == tok("ANS", add_special_tokens=False)["input_ids"]


def test_enable_thinking_changes_prefix_tokens():
    tok = FakeChatTokenizer()
    off, off_len = tokenize_pair(tok, "Q", "A", enable_thinking=False)
    on, on_len = tokenize_pair(tok, "Q", "A", enable_thinking=True)
    assert on_len > off_len
    assert off[:off_len] != on[:on_len]


def test_encode_training_sequence_does_not_joint_bpe_across_boundary():
    tok = FakeChatTokenizer()
    ids, prompt_len, rendered = encode_training_sequence(tok, "hello", "world")
    assert rendered.endswith("<|im_start|>assistant\n")
    assert ids[prompt_len:] == tok("world", add_special_tokens=False)["input_ids"]


def test_build_rows_uses_search_rendered_prompt():
    tok = FakeChatTokenizer()
    rec = {
        "problem_id": "p",
        "prompt": "Q?",
        "response": "\\boxed{1}",
        "reward": 1.0,
        "is_correct": True,
        "near_correct": False,
        "sampling_config_id": "cfg",
        "seed": 0,
        "rendered_prompt": "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n",
    }
    rows = build_training_rows([rec], tokenizer=tok, objective="graded_signed")
    assert len(rows) == 1
    prefix_ids = tok(rec["rendered_prompt"], add_special_tokens=False)["input_ids"]
    assert rows[0]["prompt_length"] == len(prefix_ids)
    assert rows[0]["input_ids"][: len(prefix_ids)] == prefix_ids
    assert rows[0]["rendered_prompt"] == rec["rendered_prompt"]


class _PrefixBackend:
    def generate(self, prompts, **kwargs):
        del kwargs
        return [
            [
                GenerationResult(
                    text="\\boxed{4}",
                    num_tokens=3,
                    extra={"rendered_prompt": f"<tmpl>{prompt}"},
                )
            ]
            for prompt in prompts
        ]


def test_search_persists_rendered_prompt(tmp_path):
    problem = Problem("add", "What is 2+2?", "4")
    settings = SearchSettings(
        initial_samples_per_config=1,
        total_samples_per_problem=1,
        seed=1,
        max_tokens=8,
    )
    result = run_search(
        [problem],
        [SamplingConfig("only", temperature=0.5, top_p=1.0)],
        _PrefixBackend(),
        MathVerifier(),
        tmp_path / "search",
        settings,
    )
    assert result["records"][0]["rendered_prompt"] == "<tmpl>What is 2+2?"
