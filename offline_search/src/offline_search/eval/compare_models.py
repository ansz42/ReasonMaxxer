from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from offline_search.eval.generate_eval import compare_eval_tables
from offline_search.utils.io import write_json


def write_comparison_table(
    named_results: dict[str, dict[str, Any]],
    output_path: str | Path,
    ks: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    if ks is None:
        ks = [1, 4, 16, 64, 256]
    table = compare_eval_tables(named_results, ks)
    write_json(output_path, table)
    return table
