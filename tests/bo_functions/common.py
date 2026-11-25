"""Shared containers and design helpers for the test-function library."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from scipy.stats import qmc
from torch import Tensor


@dataclass(frozen=True)
class BenchmarkFunction:
    """A single-fidelity, single-objective function on a box."""

    name: str
    dim: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    fn: Callable[[Tensor], Tensor]
    global_min: float | None = None
    argmin: tuple[float, ...] | None = None
    source: str = ""

    def __call__(self, x: Tensor) -> Tensor:
        return self.fn(x)

    @property
    def bounds(self) -> Tensor:
        return torch.tensor([self.lower, self.upper])


@dataclass(frozen=True)
class MultiFidelityFunction:
    """Fidelity ``levels[0]`` is the cheapest, ``levels[-1]`` the one to predict."""

    name: str
    dim: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    levels: tuple[Callable[[Tensor], Tensor], ...]
    source: str = ""
    global_min: float | None = None  # of the highest fidelity

    @property
    def num_fidelities(self) -> int:
        return len(self.levels)

    def __call__(self, x: Tensor, fidelity: int = -1) -> Tensor:
        return self.levels[fidelity](x)

    @property
    def bounds(self) -> Tensor:
        return torch.tensor([self.lower, self.upper])


@dataclass(frozen=True)
class MultiObjectiveFunction:
    """Vector-valued function (all objectives minimised) on a box."""

    name: str
    dim: int
    num_objectives: int
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    fn: Callable[[Tensor], Tensor]
    pareto_front: Callable[[int], Tensor] | None = None
    source: str = ""
    notes: str = field(default="", compare=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fn(x)

    @property
    def bounds(self) -> Tensor:
        return torch.tensor([self.lower, self.upper])


def sobol_unit(d: int, n: int, seed: int = 0, scramble: bool = True) -> Tensor:
    """``n`` Sobol points in ``[0, 1]^d`` (``n`` need not be a power of two)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.as_tensor(qmc.Sobol(d=d, scramble=scramble, seed=seed).random(n))


def sobol_design(bounds: Tensor, n: int, seed: int = 0) -> Tensor:
    """Scrambled Sobol points scaled to ``bounds`` (a ``(2, d)`` tensor)."""
    return bounds[0] + sobol_unit(bounds.shape[1], n, seed) * (bounds[1] - bounds[0])
