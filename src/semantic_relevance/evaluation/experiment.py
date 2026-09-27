"""Experiment result recording and reproducibility metadata."""

from __future__ import annotations

import json
import platform
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Serializable output from one experiment."""

    experiment: str
    metrics: dict[str, float]
    latency: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output


def runtime_metadata() -> dict[str, str]:
    """Return lightweight environment metadata for reproducibility."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def set_seed(seed: int) -> None:
    """Seed Python's standard random generator."""
    random.seed(seed)
