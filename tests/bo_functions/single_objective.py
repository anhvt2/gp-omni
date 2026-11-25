"""Single-objective benchmarks from the async/parallel BO manuscript (``asyncPar_1Feb22.tex``).

Formulas follow the manuscript and Surjanovic & Bingham's library. Where the manuscript has a
typo, the standard definition is used (noted per function). "free d" functions take ``dim``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .common import BenchmarkFunction


def _box(lo: float, hi: float, d: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (lo,) * d, (hi,) * d


def eggholder(x: Tensor) -> Tensor:
    """Restore square roots omitted in print; these reproduce the manuscript's stated optimum."""
    x1, x2 = x[:, 0], x[:, 1]
    return -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47))) - x1 * torch.sin(
        torch.sqrt(torch.abs(x1 - (x2 + 47)))
    )


def three_hump_camel(x: Tensor) -> Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    return 2 * x1**2 - 1.05 * x1**4 + x1**6 / 6 + x1 * x2 + x2**2


def six_hump_camel(x: Tensor) -> Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    return (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2


_HART_ALPHA = (1.0, 1.2, 3.0, 3.2)
_HART3_A = ((3.0, 10, 30), (0.1, 10, 35), (3.0, 10, 30), (0.1, 10, 35))
_HART3_P = ((3689, 1170, 2673), (4699, 4387, 7470), (1091, 8732, 5547), (381, 5743, 8828))
_HART6_A = (
    (10, 3, 17, 3.5, 1.7, 8),
    (0.05, 10, 17, 0.1, 8, 14),
    (3, 3.5, 1.7, 10, 17, 8),
    (17, 8, 0.05, 10, 0.1, 14),
)
_HART6_P = (
    (1312, 1696, 5569, 124, 8283, 5886),
    (2329, 4135, 8307, 3736, 1004, 9991),
    (2348, 1451, 3522, 2883, 3047, 6650),
    (4047, 8828, 8732, 5743, 1091, 381),
)


def _hartmann_inner(x: Tensor, a, p) -> Tensor:
    a, p = torch.tensor(a), 1e-4 * torch.tensor(p)
    alpha = torch.tensor(_HART_ALPHA)
    d = x.shape[1]
    inner = (a[:, :d] * (x[:, None, :] - p[:, :d]) ** 2).sum(-1)
    return (alpha * torch.exp(-inner)).sum(-1)


def hartmann3(x: Tensor) -> Tensor:
    return -_hartmann_inner(x, _HART3_A, _HART3_P)


def hartmann4(x: Tensor) -> Tensor:
    """Manuscript's sum index ``j=4..3`` is a typo; the 4-d variant uses the first 4 columns
    of the 6-d ``A``/``P`` with the ``(1.1 - sum) / 0.839`` rescale."""
    return (1.1 - _hartmann_inner(x, _HART6_A, _HART6_P)) / 0.839


def hartmann6(x: Tensor) -> Tensor:
    """Use j=1..6 and a six-dimensional box, resolving the printed reversed limits/domain."""
    return -_hartmann_inner(x, _HART6_A, _HART6_P)


def ackley(x: Tensor, a: float = 20.0, b: float = 0.2, c: float = 2 * math.pi) -> Tensor:
    return (
        -a * torch.exp(-b * torch.sqrt((x**2).mean(1)))
        - torch.exp(torch.cos(c * x).mean(1))
        + a
        + math.e
    )


def michalewicz(x: Tensor, m: float = 10.0) -> Tensor:
    i = torch.arange(1, x.shape[1] + 1)
    return -(torch.sin(x) * torch.sin(i * x**2 / math.pi) ** (2 * m)).sum(1)


def rosenbrock(x: Tensor) -> Tensor:
    return (100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2).sum(1)


def dixon_price(x: Tensor) -> Tensor:
    i = torch.arange(2, x.shape[1] + 1)
    return (x[:, 0] - 1) ** 2 + (i * (2 * x[:, 1:] ** 2 - x[:, :-1]) ** 2).sum(1)


def trid(x: Tensor) -> Tensor:
    return ((x - 1) ** 2).sum(1) - (x[:, 1:] * x[:, :-1]).sum(1)


def sum_squares(x: Tensor) -> Tensor:
    return (torch.arange(1, x.shape[1] + 1) * x**2).sum(1)


def sum_pow(x: Tensor) -> Tensor:
    return (torch.abs(x) ** torch.arange(2, x.shape[1] + 2)).sum(1)


def sphere(x: Tensor) -> Tensor:
    return (x**2).sum(1)


def rotated_hyper_ellipsoid(x: Tensor) -> Tensor:
    return torch.cumsum(x**2, dim=1).sum(1)


def perm0db(x: Tensor, beta: float = 10.0) -> Tensor:
    """Perm function 0, d, beta (``beta=10`` as in Surjanovic & Bingham; unspecified in the
    manuscript)."""
    d = x.shape[1]
    j = torch.arange(1, d + 1, dtype=x.dtype)  # float: integer j**i overflows for large d
    total = 0.0
    for i in range(1, d + 1):
        total = total + ((j + beta) * (x**i - 1.0 / j**i)).sum(1) ** 2
    return total


def _fixed(name, dim, lo, hi, fn, gmin, argmin, source="asyncPar_1Feb22 / sfu.ca"):
    return BenchmarkFunction(name, dim, lo, hi, fn, gmin, argmin, source)


def fixed_dim_functions() -> dict[str, BenchmarkFunction]:
    """The functions with a fixed dimension in the manuscript."""
    return {
        f.name: f
        for f in (
            _fixed("eggholder", 2, *_box(-512, 512, 2), eggholder, -959.6407, (512.0, 404.2319)),
            _fixed("three_hump_camel", 2, *_box(-5, 5, 2), three_hump_camel, 0.0, (0.0, 0.0)),
            _fixed(
                "six_hump_camel", 2, (-3, -2), (3, 2), six_hump_camel, -1.0316, (-0.0898, 0.7126)
            ),
            _fixed(
                "hartmann3", 3, *_box(0, 1, 3), hartmann3, -3.86278, (0.114614, 0.555649, 0.852547)
            ),
            _fixed("hartmann4", 4, *_box(0, 1, 4), hartmann4, -3.135474, None),
            _fixed(
                "hartmann6",
                6,
                *_box(0, 1, 6),
                hartmann6,
                -3.32237,
                (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
            ),
        )
    }


def free_dim_function(name: str, dim: int) -> BenchmarkFunction:
    """A "free d" benchmark at the requested dimension."""
    d = dim
    ones, zeros = (1.0,) * d, (0.0,) * d
    table = {
        "ackley": (*_box(-32.768, 32.768, d), ackley, 0.0, zeros),
        "michalewicz": (*_box(0, math.pi, d), michalewicz, None, None),
        "rosenbrock": (*_box(-5, 10, d), rosenbrock, 0.0, ones),
        "dixon_price": (
            *_box(-10, 10, d),
            dixon_price,
            0.0,
            tuple(2.0 ** (-(2.0**i - 2) / 2.0**i) for i in range(1, d + 1)),
        ),
        "trid": (
            *_box(-(d**2), d**2, d),
            trid,
            -d * (d + 4) * (d - 1) / 6,
            tuple(float(i * (d + 1 - i)) for i in range(1, d + 1)),
        ),
        "sum_squares": (*_box(-5.12, 5.12, d), sum_squares, 0.0, zeros),
        "sum_pow": (*_box(-1, 1, d), sum_pow, 0.0, zeros),
        "sphere": (*_box(-5.12, 5.12, d), sphere, 0.0, zeros),
        "rotated_hyper_ellipsoid": (*_box(-65.536, 65.536, d), rotated_hyper_ellipsoid, 0.0, zeros),
        "perm0db": (
            *_box(-d, d, d),
            perm0db,
            0.0,
            tuple(1.0 / j for j in range(1, d + 1)),
        ),
    }
    lo, hi, fn, gmin, argmin = table[name]
    return _fixed(name, d, lo, hi, fn, gmin, argmin)


FREE_DIM_NAMES = (
    "ackley",
    "michalewicz",
    "rosenbrock",
    "dixon_price",
    "trid",
    "sum_squares",
    "sum_pow",
    "sphere",
    "rotated_hyper_ellipsoid",
    "perm0db",
)

# Manuscript dimensions for the "free d" functions (async BO paper), used for optimum checks.
MANUSCRIPT_DIM = {
    "ackley": 5,
    "michalewicz": 10,
    "rosenbrock": 20,
    "dixon_price": 25,
    "trid": 30,
    "sum_squares": 40,
    "sum_pow": 50,
    "sphere": 60,
    "rotated_hyper_ellipsoid": 70,
    "perm0db": 80,
}
