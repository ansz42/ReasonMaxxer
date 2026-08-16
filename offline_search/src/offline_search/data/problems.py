from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from offline_search.prompting import build_user_prompt
from offline_search.search.search_runner import Problem


def load_problems_file(path: str | Path, *, prompt_style: str = "qwen3_chat") -> list[Problem]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("records", payload) if isinstance(payload, (dict, list)) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Unsupported problems file: {path}")
    problems: list[Problem] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        text = row.get("problem_text") or row.get("problem") or row.get("prompt")
        if text is None:
            continue
        pid = str(row.get("problem_id", f"problem/{i}"))
        reference = row.get("ground_truth")
        if reference is None:
            reference = row.get("reference_answer")
        if reference is None:
            reference = row.get("answer")
        prompt = str(row.get("prompt") or build_user_prompt(str(text), prompt_style=prompt_style))
        problems.append(
            Problem(
                problem_id=pid,
                prompt=prompt,
                reference_answer=None if reference is None else str(reference),
                extra={k: v for k, v in row.items() if k not in {"problem_id", "prompt"}},
            )
        )
    if not problems:
        raise ValueError(f"No usable problems in {path}")
    return problems
