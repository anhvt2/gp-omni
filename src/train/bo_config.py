"""Common configuration for Bayesian optimisation loops."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("bo")


known_constraint_fn = Callable[[np.ndarray], float]


def check_constraint(x: np.ndarray) -> float:
    """Accept every point by default; supply a callable for known constraints."""
    return 1.0


@dataclass(frozen=True)
class batch_settings:
    """``batch_settings.dat``: how many jobs of each kind may be in flight."""

    exploit: int
    explore: int
    explore_classifier: int

    @property
    def total(self) -> int:
        return self.exploit + self.explore + self.explore_classifier


@dataclass
class bo_config:
    """Shared Bayesian optimisation settings.

    ``max_iter`` counts evaluations for async BO and iterations for batch-sequential BO.
    """

    model_name: str
    x_lb: np.ndarray
    x_ub: np.ndarray
    exploit_size: int = DEFAULTS["loop"][
        "exploit_size"
    ]  # acquisition jobs per batch / concurrent slots
    explore_size: int = DEFAULTS["loop"]["explore_size"]  # objective-model exploration jobs
    explore_classifier_size: int = DEFAULTS["loop"][
        "explore_classifier_size"
    ]  # constraint-classifier exploration jobs
    max_iter: int = DEFAULTS["loop"][
        "max_iter"
    ]  # includes initial points; units depend on the loop
    num_init_point: int = DEFAULTS["loop"]["num_init_point"]
    x_scale: np.ndarray | float = DEFAULTS["loop"][
        "x_scale"
    ]  # rough scale so scaled inputs are O(1)
    y_scale: float = DEFAULTS["loop"]["y_scale"]
    check_time: float = DEFAULTS["loop"][
        "check_time"
    ]  # minutes between polls while waiting for jobs
    wait_time: float = DEFAULTS["loop"][
        "wait_time"
    ]  # hours before an unfinished job is declared failed
    gp_max_iter: int = DEFAULTS["loop"][
        "gp_max_iter"
    ]  # hyper-parameter optimiser iterations per fit
    cma_max_evals: int = DEFAULTS["loop"][
        "cma_max_evals"
    ]  # CMA-ES budget per acquisition proposal (explore uses 100)
    seed: int | None = DEFAULTS["loop"]["seed"]
    query_script: str | None = DEFAULTS["loop"][
        "query_script"
    ]  # bash script run inside the job folder (async by itself)
    known_constraint: known_constraint_fn = field(default=check_constraint)

    def __post_init__(self):
        self.x_lb, self.x_ub = np.asarray(self.x_lb, float), np.asarray(self.x_ub, float)
        self.x_scale = np.broadcast_to(np.asarray(self.x_scale, float), self.x_lb.shape).copy()

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> Self:
        """Load settings from a JSON file.

        Keys are the field names; ``overrides`` (e.g. ``known_constraint=...``, which cannot
        live in JSON) take precedence over the file.
        """
        values = json.loads(Path(path).read_text())
        return cls(**{**values, **overrides})

    def to_json(self, path: str | Path) -> None:
        """Write the JSON-representable settings (everything but ``known_constraint``)."""
        values = {k: v for k, v in self.__dict__.items() if k != "known_constraint"}
        values = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in values.items()}
        Path(path).write_text(json.dumps(values, indent=2) + "\n")

    @property
    def batch(self) -> batch_settings:
        return batch_settings(self.exploit_size, self.explore_size, self.explore_classifier_size)
