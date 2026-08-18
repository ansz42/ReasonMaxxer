from __future__ import annotations

from offline_search.search.clipping import clip_retry_seed, is_clipped, is_clipped_record
from offline_search.search.generate import GenerationResult


def test_length_finish_reason_is_clipped():
    result = GenerationResult(text="partial", num_tokens=10, finish_reason="length")
    assert is_clipped(result, max_tokens=128) is True


def test_stop_finish_reason_is_not_clipped():
    result = GenerationResult(text="ok", num_tokens=10, finish_reason="stop")
    assert is_clipped(result, max_tokens=128) is False


def test_token_count_at_cap_is_clipped_without_finish_reason():
    result = GenerationResult(text="x", num_tokens=16, finish_reason=None)
    assert is_clipped(result, max_tokens=16) is True
    assert is_clipped(result, max_tokens=17) is False


def test_retry_seed_differs_from_original_and_is_stable():
    first = clip_retry_seed(42)
    assert first != 42
    assert clip_retry_seed(42) == first
    assert clip_retry_seed(43) != first


def test_record_is_clipped_by_finish_reason_tokens_or_discard_flag():
    assert is_clipped_record({"finish_reason": "length"}) is True
    assert is_clipped_record({"generated_tokens": 3000}, max_tokens=3000) is True
    assert is_clipped_record({"discarded": True}) is True
    assert is_clipped_record({"generated_tokens": 10, "finish_reason": "stop"}, max_tokens=3000) is False
    assert is_clipped_record({"generated_tokens": 3000}) is False
