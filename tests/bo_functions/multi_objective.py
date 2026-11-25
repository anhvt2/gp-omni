"""ZDT and DTLZ suites from the multi-objective BO manuscripts (``sMoConsBO_25Aug21.tex``).

Preserve the manuscript's 0.5 scale for DTLZ2-6 and its partial alpha power in DTLZ4.
DTLZ4 is a well-defined variant, but its front is not the standard spherical front.
DTLZ1 follows the manuscript's explicit tail-variable convention (k = d - M + 1).
DTLZ5/6 use the cited DTLZ angle convention to resolve undefined theta_1 and duplicate
pi/2 scaling in the printed equations. See the audit in the generated accuracy report.

BoTorch cross-check (``tests/test_bo_functions.py``): ZDT1/2 and DTLZ1 agree exactly; DTLZ2/3/5
agree after applying the manuscript's 0.5 scale. BoTorch's
ZDT3 returns ``h`` instead of ``g * h`` in the installed version, so ZDT3 and the manuscript's
DTLZ4 variant are checked against their analytic definitions instead.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .common import MultiObjectiveFunction, sobol_unit

SOURCE = "sMoConsBO_25Aug21"
ZDT3_SEGMENTS = (
    (0.0, 0.0830015349),
    (0.1822287280, 0.2577623634),
    (0.4093136748, 0.4538821041),
    (0.6183967944, 0.6525117038),
    (0.8233317983, 0.8518328654),
)


# ------------------------------------------------------------------------------------- ZDT
def _zdt_g(x: Tensor) -> Tensor:
    return 1 + 9 * x[:, 1:].sum(1) / (x.shape[1] - 1)


def zdt1(x: Tensor) -> Tensor:
    f1, g = x[:, 0], _zdt_g(x)
    return torch.stack([f1, g * (1 - torch.sqrt(f1 / g))], 1)


def zdt2(x: Tensor) -> Tensor:
    f1, g = x[:, 0], _zdt_g(x)
    return torch.stack([f1, g * (1 - (f1 / g) ** 2)], 1)


def zdt3(x: Tensor) -> Tensor:
    f1, g = x[:, 0], _zdt_g(x)
    h = 1 - torch.sqrt(f1 / g) - (f1 / g) * torch.sin(10 * math.pi * f1)
    return torch.stack([f1, g * h], 1)


def zdt4(x: Tensor) -> Tensor:
    f1, d = x[:, 0], x.shape[1]
    g = 1 + 10 * (d - 1) + (x[:, 1:] ** 2 - 10 * torch.cos(4 * math.pi * x[:, 1:])).sum(1)
    return torch.stack([f1, g * (1 - torch.sqrt(f1 / g))], 1)


def zdt6(x: Tensor) -> Tensor:
    f1 = 1 - torch.exp(-4 * x[:, 0]) * torch.sin(6 * math.pi * x[:, 0]) ** 6
    g = 1 + 9 * (x[:, 1:].sum(1) / (x.shape[1] - 1)) ** 0.25
    return torch.stack([f1, g * (1 - (f1 / g) ** 2)], 1)


def _front(f2_of_f1, lo: float = 0.0, hi: float = 1.0):
    def front(n: int) -> Tensor:
        f1 = torch.linspace(lo, hi, n)
        return torch.stack([f1, f2_of_f1(f1)], 1)

    return front


def _zdt3_front(n: int) -> Tensor:
    if n < 0:
        raise ValueError("n must be nonnegative")
    per, remainder = divmod(n, len(ZDT3_SEGMENTS))
    f1 = torch.cat([
        torch.linspace(a, b, per + (i < remainder))
        for i, (a, b) in enumerate(ZDT3_SEGMENTS)
    ])
    return torch.stack([f1, 1 - torch.sqrt(f1) - f1 * torch.sin(10 * math.pi * f1)], 1)


def zdt(name: str, dim: int = 3) -> MultiObjectiveFunction:
    """``zdt1`` .. ``zdt4``, ``zdt6`` with ``dim`` inputs (manuscript uses 3)."""
    fns = {"zdt1": zdt1, "zdt2": zdt2, "zdt3": zdt3, "zdt4": zdt4, "zdt6": zdt6}
    fronts = {
        "zdt1": _front(lambda f: 1 - torch.sqrt(f)),
        "zdt2": _front(lambda f: 1 - f**2),
        "zdt3": _zdt3_front,
        "zdt4": _front(lambda f: 1 - torch.sqrt(f)),
        "zdt6": _front(lambda f: 1 - f**2, 0.2807753191, 1.0),
    }
    lower = (0.0,) * dim
    upper = (1.0,) + ((5.0,) * (dim - 1) if name == "zdt4" else (1.0,) * (dim - 1))
    if name == "zdt4":
        lower = (0.0,) + (-5.0,) * (dim - 1)
    return MultiObjectiveFunction(name, dim, 2, lower, upper, fns[name], fronts[name], SOURCE)


# ------------------------------------------------------------------------------------ DTLZ
def _split(x: Tensor, m: int) -> tuple[Tensor, Tensor]:
    return x[:, : m - 1], x[:, m - 1 :]


def _angles_to_objectives(g: Tensor, theta: Tensor, m: int) -> Tensor:
    """``f_i = 0.5*(1+g) * prod_{j<M-i} cos(theta_j) * sin(theta_{M-i})`` (0-based, ``theta`` already
    multiplied by pi/2), the spherical-coordinate form shared by DTLZ2-6."""
    fs = []
    for i in range(m):
        f = 1 + g
        f = f * torch.cos(theta[:, : m - 1 - i]).prod(1)
        if i > 0:
            f = f * torch.sin(theta[:, m - 1 - i])
        fs.append(f)
    return 0.5 * torch.stack(fs, 1)


def dtlz1(x: Tensor, m: int = 3) -> Tensor:
    head, tail = _split(x, m)
    k = tail.shape[1]
    g = 100 * (k + ((tail - 0.5) ** 2 - torch.cos(20 * math.pi * (tail - 0.5))).sum(1))
    fs = []
    for i in range(m):
        f = 0.5 * (1 + g) * head[:, : m - 1 - i].prod(1)
        if i > 0:
            f = f * (1 - head[:, m - 1 - i])
        fs.append(f)
    return torch.stack(fs, 1)


def dtlz2(x: Tensor, m: int = 3) -> Tensor:
    head, tail = _split(x, m)
    return _angles_to_objectives(((tail - 0.5) ** 2).sum(1), head * math.pi / 2, m)


def dtlz3(x: Tensor, m: int = 3) -> Tensor:
    head, tail = _split(x, m)
    k = tail.shape[1]
    g = 100 * (k + ((tail - 0.5) ** 2 - torch.cos(20 * math.pi * (tail - 0.5))).sum(1))
    return _angles_to_objectives(g, head * math.pi / 2, m)


def dtlz4(x: Tensor, m: int = 3, alpha: float = 100.0) -> Tensor:
    """Printed variant: alpha applies to cosine factors only (100 is an assumed default)."""
    head, tail = _split(x, m)
    g = ((tail - 0.5) ** 2).sum(1)
    fs = []
    for i in range(m):
        f = 0.5 * (1 + g) * torch.cos(head[:, : m - 1 - i] ** alpha * math.pi / 2).prod(1)
        if i > 0:
            f = f * torch.sin(head[:, m - 1 - i] * math.pi / 2)
        fs.append(f)
    return torch.stack(fs, 1)


def _degenerate_theta(head: Tensor, g: Tensor) -> Tensor:
    """DTLZ5/6: ``theta_1 = x_1`` and ``theta_i = pi (1 + 2 g x_i) / (4 (1 + g))`` otherwise,
    returned already scaled (theta_i here is the angle itself, not a multiple of pi/2)."""
    theta = math.pi / (4 * (1 + g[:, None])) * (1 + 2 * g[:, None] * head)
    return torch.cat([head[:, :1] * math.pi / 2, theta[:, 1:]], 1)


def dtlz5(x: Tensor, m: int = 3) -> Tensor:
    head, tail = _split(x, m)
    g = ((tail - 0.5) ** 2).sum(1)
    return _angles_to_objectives(g, _degenerate_theta(head, g), m)


def dtlz6(x: Tensor, m: int = 3) -> Tensor:
    head, tail = _split(x, m)
    g = (tail**0.1).sum(1)
    return _angles_to_objectives(g, _degenerate_theta(head, g), m)


def _sphere_front(m: int):
    """Radius-0.5 sphere front (positive orthant), as ``n`` Sobol-distributed points."""

    def front(n: int) -> Tensor:
        u = sobol_unit(m - 1, n, scramble=False) * math.pi / 2
        return _angles_to_objectives(torch.zeros(n), u, m)

    return front


def _simplex_front(m: int):
    """DTLZ1: ``sum_i f_i = 0.5``, sampled uniformly from the simplex."""

    def front(n: int) -> Tensor:
        u = sobol_unit(m, n, scramble=False).clamp_min(1e-12)
        e = -torch.log(u)
        return 0.5 * e / e.sum(1, keepdim=True)

    return front


def dtlz(name: str, num_objectives: int = 3, dim: int = 4) -> MultiObjectiveFunction:
    """``dtlz1`` .. ``dtlz6``; the manuscript uses ``M = 3`` objectives and ``d = 4`` inputs."""
    m = num_objectives
    fns = {"dtlz1": dtlz1, "dtlz2": dtlz2, "dtlz3": dtlz3, "dtlz4": dtlz4, "dtlz5": dtlz5}
    fns["dtlz6"] = dtlz6
    fronts = {
        "dtlz1": _simplex_front(m),
        "dtlz2": _sphere_front(m),
        "dtlz3": _sphere_front(m),
        "dtlz4": None,  # partial alpha powers do not preserve the spherical identity
        "dtlz5": None,
        "dtlz6": None,
    }
    fn = fns[name]
    return MultiObjectiveFunction(
        name,
        dim,
        m,
        (0.0,) * dim,
        (1.0,) * dim,
        lambda x: fn(x, m),
        fronts[name],
        SOURCE,
    )


ZDT_NAMES = ("zdt1", "zdt2", "zdt3", "zdt4", "zdt6")
DTLZ_NAMES = ("dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6")


def all_multi_objective() -> list[MultiObjectiveFunction]:
    return [zdt(n) for n in ZDT_NAMES] + [dtlz(n) for n in DTLZ_NAMES]
