from __future__ import annotations

from typing import Any


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
