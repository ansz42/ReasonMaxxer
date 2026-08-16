from __future__ import annotations

from typing import Any


def uses_chat_template(tokenizer: Any) -> bool:
    return tokenizer is not None and hasattr(tokenizer, "apply_chat_template")


def build_user_prompt(problem_text: str, *, prompt_style: str = "qwen3_chat") -> str:
    style = (prompt_style or "qwen3_chat").strip().lower()
    if style == "raw":
        return problem_text
    if style == "qwen_boxed":
        return (
            "Solve the following math problem step by step. Put your final answer in \\boxed{}.\n\n"
            f"Problem: {problem_text}\n\n"
            "Solution:"
        )
    if style in {"qwen3_chat", "chat_template"}:
        return (
            "Solve the following math problem step by step. "
            "Give a concise solution and put your final answer in \\boxed{}.\n\n"
            f"Problem: {problem_text}"
        )
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def apply_chat_template(
    tokenizer: Any,
    user_text: str,
    *,
    enable_thinking: bool = False,
) -> str:
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return user_text
    messages = [{"role": "user", "content": user_text}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": bool(enable_thinking),
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        if not enable_thinking:
            messages = [{"role": "user", "content": "/no_think\n" + user_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def render_generation_prompt(
    tokenizer: Any,
    user_text: str,
    *,
    enable_thinking: bool = False,
) -> str:
    """Same string prefix TransformersBackend feeds the model before decoding."""
    return apply_chat_template(tokenizer, user_text, enable_thinking=enable_thinking)


def encode_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    if tokenizer is None:
        raise ValueError("tokenizer is required")
    if callable(tokenizer) and not isinstance(tokenizer, type):
        try:
            out = tokenizer(text, add_special_tokens=add_special_tokens)
        except TypeError:
            out = tokenizer(text)
        ids = out["input_ids"] if isinstance(out, dict) else out
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        except TypeError:
            ids = tokenizer.encode(text)
        return [int(x) for x in ids]
    raise TypeError(f"Tokenizer {type(tokenizer)!r} has no encode path")


def encode_generation_prefix(
    tokenizer: Any,
    user_text: str,
    *,
    enable_thinking: bool = False,
    rendered_prefix: str | None = None,
) -> tuple[str, list[int]]:
    """Tokenize the exact chat-template prefix used at generation time.

    Chat-templated strings already contain special tokens, so encoding uses
    add_special_tokens=False to avoid a second BOS and to stay aligned with
    TransformersBackend.generate.
    """
    if rendered_prefix is not None:
        rendered = str(rendered_prefix)
    elif uses_chat_template(tokenizer):
        rendered = render_generation_prompt(tokenizer, user_text, enable_thinking=enable_thinking)
    else:
        rendered = user_text
    add_special = not uses_chat_template(tokenizer)
    return rendered, encode_ids(tokenizer, rendered, add_special_tokens=add_special)


def encode_training_sequence(
    tokenizer: Any,
    user_text: str,
    response: str,
    *,
    enable_thinking: bool = False,
    rendered_prefix: str | None = None,
) -> tuple[list[int], int, str]:
    """Prefix tokens + separately encoded assistant response tokens.

    Do not encode `prefix + response` as one string: that is a different
    conditional than the one the model sampled from.
    """
    rendered, prefix_ids = encode_generation_prefix(
        tokenizer,
        user_text,
        enable_thinking=enable_thinking,
        rendered_prefix=rendered_prefix,
    )
    add_special = not uses_chat_template(tokenizer)
    response_ids = encode_ids(tokenizer, response, add_special_tokens=add_special)
    return prefix_ids + response_ids, len(prefix_ids), rendered
