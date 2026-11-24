"""Load the JSON settings shipped in ``config/`` with the installed package.

Defaults are read when their consuming module is imported. Explicit API arguments
always take precedence. Resource access works from source, wheels, and zip imports.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def load_defaults(name: str) -> dict[str, Any]:
    """Return a fresh copy of a named, packaged defaults document."""
    if name not in {"models", "training", "inference", "builders", "bo"}:
        raise ValueError(f"unknown defaults document: {name!r}")
    return json.loads(files("gp_omni.config").joinpath(f"{name}.json").read_text())


def load_model_config(path: str | Path, **overrides: Any) -> dict[str, Any]:
    """Read a flat constructor settings file; explicit overrides take precedence."""
    values = json.loads(Path(path).read_text())
    if not isinstance(values, dict):
        raise TypeError("model configuration must be a JSON object")
    return {**values, **overrides}
