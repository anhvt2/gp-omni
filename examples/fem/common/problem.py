"""A parametric design problem: bounded variables, one objective and limits on metrics."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import getfile
from pathlib import Path

import numpy as np

metrics_map = Mapping[str, float]


@dataclass(frozen=True)
class variable:
    name: str
    lower: float
    upper: float
    reference: float


@dataclass(frozen=True)
class constraint:
    metric: str
    sense: str  # "<=" or ">="
    limit: float

    def satisfied(self, metrics: metrics_map) -> bool:
        value = metrics[self.metric]
        return bool(value <= self.limit if self.sense == "<=" else value >= self.limit)


def at_most(metric: str, limit: float) -> constraint:
    return constraint(metric, "<=", limit)


def at_least(metric: str, limit: float) -> constraint:
    return constraint(metric, ">=", limit)


@dataclass(frozen=True)
class design_problem:
    """Minimize ``objective`` subject to ``constraints``.

    ``solve(design, mesh_scale)`` returns the metrics of one design; ``mesh_scale``
    multiplies every element size, so values below one refine the mesh.
    """

    name: str
    title: str
    variables: tuple[variable, ...]
    objective: str
    constraints: tuple[constraint, ...]
    solve: Callable[[dict[str, float], float], dict[str, float]]

    @property
    def source(self) -> Path:
        """The case script that defines ``solve``."""
        return Path(getfile(self.solve))

    def reference_design(self) -> dict[str, float]:
        return {v.name: v.reference for v in self.variables}

    def design_from_unit(self, x: np.ndarray) -> dict[str, float]:
        """Map a point of the unit cube onto the variable bounds."""
        if len(x) != len(self.variables):
            raise ValueError(f"expected {len(self.variables)} values, got {len(x)}")
        return {
            v.name: float(np.clip(v.lower + u * (v.upper - v.lower), v.lower, v.upper))
            for v, u in zip(self.variables, x)
        }

    def analyze(self, design: dict[str, float], mesh_scale: float = 1.0) -> dict[str, float]:
        if set(design) != {v.name for v in self.variables}:
            raise ValueError(f"{self.name} needs exactly the variables of its definition")
        for v in self.variables:
            if not v.lower <= design[v.name] <= v.upper:
                raise ValueError(f"{v.name} must lie in [{v.lower}, {v.upper}]")
        if not mesh_scale > 0:
            raise ValueError("mesh_scale must be positive")
        return self.solve(dict(design), mesh_scale)

    def is_feasible(self, metrics: metrics_map) -> bool:
        return all(c.satisfied(metrics) for c in self.constraints)
