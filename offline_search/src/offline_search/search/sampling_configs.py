from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SamplingConfig:
    config_id: str
    temperature: float
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SEARCH_CONFIG_SPECS: list[dict[str, Any]] = [
    {"temperature": 0.35, "top_p": 0.90},
    {"temperature": 0.50, "top_p": 0.95},
    {"temperature": 0.70, "top_p": 0.95},
    {"temperature": 0.80, "top_p": 1.00},
    {"temperature": 1.00, "top_p": 0.95},
    {"temperature": 1.15, "top_p": 0.98},
    {"temperature": 1.30, "top_p": 1.00},
    {"temperature": 0.85, "top_p": 0.98, "repetition_penalty": 1.05},
]


def _config_id(spec: dict[str, Any], index: int) -> str:
    if spec.get("config_id"):
        return str(spec["config_id"])
    parts = [f"t{float(spec.get('temperature', 1.0)):.2f}", f"p{float(spec.get('top_p', 1.0)):.2f}"]
    if spec.get("top_k") is not None:
        parts.append(f"k{int(spec['top_k'])}")
    if float(spec.get("repetition_penalty", 1.0)) != 1.0:
        parts.append(f"rp{float(spec['repetition_penalty']):.2f}")
    return f"cfg{index:02d}_" + "_".join(parts)


def configs_from_specs(specs: Iterable[dict[str, Any]]) -> list[SamplingConfig]:
    out: list[SamplingConfig] = []
    for i, spec in enumerate(specs):
        out.append(
            SamplingConfig(
                config_id=_config_id(spec, i),
                temperature=float(spec.get("temperature", 1.0)),
                top_p=float(spec.get("top_p", 1.0)),
                top_k=int(spec["top_k"]) if spec.get("top_k") is not None else None,
                repetition_penalty=float(spec.get("repetition_penalty", 1.0)),
            )
        )
    return out


def default_search_configs() -> list[SamplingConfig]:
    return configs_from_specs(DEFAULT_SEARCH_CONFIG_SPECS)
