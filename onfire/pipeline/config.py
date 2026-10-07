"""Parameter Config loading and defaults.

The Parameter Config is a transparent YAML file holding algorithmic parameter
overrides and pipeline control options, kept separate from the Frame Table as
recorded in ADR 0002.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Pipeline defaults applied before any user YAML overrides.
DEFAULT_CONFIG: dict[str, Any] = {
    "mode": "pair_difference",
    "frames": "frames.ascii",
    "output_dir": "checkpoints",
    "qa_dir": "qa",
    "steps": {},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto an independent copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML Parameter Config merged onto :data:`DEFAULT_CONFIG`.

    Args:
        path: Path to a YAML file, or ``None`` to return the defaults.

    Returns:
        The resolved Parameter Config as a nested dictionary.

    Raises:
        FileNotFoundError: If ``path`` is provided but does not exist.
        TypeError: If the YAML document is not a mapping.
    """
    if path is None:
        return _merge(DEFAULT_CONFIG, {})
    text = Path(path).read_text()
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Parameter Config must be a YAML mapping, got {type(data).__name__}")
    return _merge(DEFAULT_CONFIG, data)
