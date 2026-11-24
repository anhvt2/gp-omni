"""Common configuration for Bayesian optimisation loops."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

KnownConstraint = Callable[[np.ndarray], float]


def check_constraint(x: np.ndarray) -> float:
    """Accept every point by default; supply a callable for known constraints."""
    return 1.0


@dataclass(frozen=True)
class BatchSettings:
    """``batch_settings.dat``: how many jobs of each kind may be in flight."""

    exploit: int
    explore: int
    explore_classifier: int

    @property
    def total(self) -> int:
        return self.exploit + self.explore + self.explore_classifier


@dataclass
class BOConfig:
    """Shared Bayesian optimisation settings.

    ``max_iter`` counts evaluations for async BO and iterations for batch-sequential BO.
    """

    model_name: str
    x_lb: np.ndarray
    x_ub: np.ndarray
    exploit_size: int = 3  # acquisition jobs per batch / concurrent slots
    explore_size: int = 2  # objective-model exploration jobs
    explore_classifier_size: int = 1  # constraint-classifier exploration jobs
    max_iter: int = 400  # includes initial points; units depend on the loop
    num_init_point: int = 2
    x_scale: np.ndarray | float = 1.0  # rough scale so scaled inputs are O(1)
    y_scale: float = 1.0
    check_time: float = 0.10  # minutes between polls while waiting for jobs
    wait_time: float = 1.0  # hours before an unfinished job is declared failed
    gp_max_iter: int = 200  # hyper-parameter optimiser iterations per fit
    cma_max_evals: int = 250  # CMA-ES budget per acquisition proposal (explore uses 100)
    seed: int | None = None
    query_script: str | None = None  # bash script run inside the job folder (async by itself)
    known_constraint: KnownConstraint = field(default=check_constraint)

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
    def batch(self) -> BatchSettings:
        return BatchSettings(self.exploit_size, self.explore_size, self.explore_classifier_size)
