"""What a CFD design case is: variables, an objective, constraints and a case builder."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class variable:
    name: str
    lower: float
    upper: float
    reference: float


@dataclass(frozen=True)
class limit:
    """A metric that may worsen by ``margin`` (a fraction) relative to the reference design."""

    metric: str
    sense: str  # "<=" or ">="
    margin: float = 0.1

    def bound(self, reference: float) -> float:
        slack = self.margin * abs(reference)
        return reference - slack if self.sense == ">=" else reference + slack

    def satisfied(self, metric: float, reference: float) -> bool:
        bound = self.bound(reference)
        return bool(metric >= bound if self.sense == ">=" else metric <= bound)


def at_least(metric: str, margin: float = 0.1) -> limit:
    return limit(metric, ">=", margin)


def at_most(metric: str, margin: float = 0.1) -> limit:
    return limit(metric, "<=", margin)


@dataclass(frozen=True)
class mesh_level:
    """``resolution`` scales the cell count in each direction."""

    name: str
    resolution: float
    iterations: int
    time_steps: int

    def cells(self, base: int, minimum: int = 1) -> int:
        return max(minimum, round(base * self.resolution))


SMOKE = mesh_level("smoke", 1.0, 200, 50)
STANDARD = mesh_level("standard", 2.0, 2000, 2000)
LEVELS = {level.name: level for level in (SMOKE, STANDARD)}


@dataclass
class run_plan:
    """Commands that mesh and solve a prepared case, and the function that reads results.

    ``measure`` may return ``flow_imbalance``, which the pipeline checks.
    """

    solve: list[list[str]]
    measure: Callable[[Path], dict[str, float]]
    mesh: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class cfd_case:
    name: str
    family: str
    title: str
    mesher: str  # "structured", "snappy" or "bundled"
    solver: str
    variables: tuple[variable, ...]
    objective: str
    constraints: tuple[limit, ...]
    build: Callable[[Path, dict[str, float], mesh_level], run_plan]
    maximize: bool = False
    positive: tuple[str, ...] = ()  # metrics a physical result keeps above zero
    simplification: str = ""

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

    def check_design(self, design: Mapping[str, float]) -> None:
        if set(design) != {v.name for v in self.variables}:
            raise ValueError(f"{self.name} needs exactly the variables of its definition")
        for v in self.variables:
            if not v.lower <= design[v.name] <= v.upper:
                raise ValueError(f"{v.name} must lie in [{v.lower}, {v.upper}]")

    def feasible(self, metrics: Mapping[str, float], reference: Mapping[str, float]) -> bool:
        return all(c.satisfied(metrics[c.metric], reference[c.metric]) for c in self.constraints)

    def flags(self, metrics: Mapping[str, float]) -> list[str]:
        """Implausible values worth a look; they do not fail a smoke run."""
        found = [f"{n} is not positive" for n in self.positive if not metrics[n] > 0]
        for name, number in metrics.items():
            if name.endswith("efficiency") and not 0 < number < 1:
                found.append(f"{name} is outside (0, 1)")
        return found
