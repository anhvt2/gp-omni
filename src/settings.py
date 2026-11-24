"""Load the JSON settings shipped in ``config/`` with the installed package.

Defaults are read when their consuming module is imported. Explicit API arguments
always take precedence. Resource access works from source, wheels, and zip imports.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_defaults(name: str) -> dict[str, Any]:
    """Return a fresh copy of a named, packaged defaults document."""
    if name not in {"inference", "models", "training"}:
        raise ValueError(f"unknown defaults document: {name!r}")
    return json.loads(files("gp_omni.config").joinpath(f"{name}.json").read_text())
