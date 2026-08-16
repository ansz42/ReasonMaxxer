from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = file_path.with_suffix(file_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(file_path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parquet_cell(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return [_parquet_cell(v) for v in value]
    return value


def records_to_parquet(path: str | Path, records: Sequence[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to write parquet") from exc

    if not records:
        table = pa.table({})
        pq.write_table(table, file_path)
        return

    sanitized = [{k: _parquet_cell(v) for k, v in record.items()} for record in records]
    table = pa.Table.from_pylist(sanitized)
    pq.write_table(table, file_path)


def records_from_parquet(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return table.to_pylist()
