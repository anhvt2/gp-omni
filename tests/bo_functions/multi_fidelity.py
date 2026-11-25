"""Multi-fidelity functions from ``sMfConsBO_7Mar20.tex`` (final submission).

Levels are ordered cheapest first, so ``levels[-1]`` is the high-fidelity function ``f_H``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .common import MultiFidelityFunction

SOURCE = "sMfConsBO_7Mar20"


# ------------------------------------------------------------------ Forrester (1d, 2 levels)
def forrester_high(x: Tensor) -> Tensor:
    return (6 * x[:, 0] - 2) ** 2 * torch.sin(12 * x[:, 0] - 4)


def forrester_low(x: Tensor) -> Tensor:
    return 0.5 * forrester_high(x) + 10 * (x[:, 0] - 0.5) - 5


def forrester() -> MultiFidelityFunction:
    return MultiFidelityFunction(
        "forrester", 1, (0.0,), (1.0,), (forrester_low, forrester_high), SOURCE, -6.02074
    )


# --------------------------------------------------------------------- Currin (2d, 2 levels)
def currin_high(x: Tensor) -> Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    num = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
    den = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
    # exp(-1/(2 x2)) -> 0 as x2 -> 0+; avoid 0/0 at the boundary.
    shrink = 1 - torch.exp(-1 / (2 * x2.clamp_min(1e-300)))
    return shrink * num / den


def currin_low(x: Tensor) -> Tensor:
    x1, x2 = x[:, 0], x[:, 1]

    def shifted(dx1: float, dx2: Tensor) -> Tensor:
        return currin_high(torch.stack([x1 + dx1, dx2], dim=1))

    up, down = x2 + 0.05, (x2 - 0.05).clamp_min(0.0)
    return 0.25 * (shifted(0.05, up) + shifted(0.05, down)) + 0.25 * (
        shifted(-0.05, up) + shifted(-0.05, down)
    )


def currin() -> MultiFidelityFunction:
    return MultiFidelityFunction(
        "currin", 2, (0.0, 0.0), (1.0, 1.0), (currin_low, currin_high), SOURCE
    )


# -------------------------------------------------------------------- Borehole (8d, 2/3 levels)
BOREHOLE_LOWER = (0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0)
BOREHOLE_UPPER = (0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0)


def _borehole(x: Tensor, num: float, shift: float) -> Tensor:
    """Manuscript's ``x_1..x_8`` = (r_w, r, T_u, H_u, T_l, H_l, L, K_w)."""
    x1, x2, x3, x4, x5, x6, x7, x8 = x.unbind(1)
    log_term = torch.log(x2 / x1)
    return (num * x3 * (x4 - x6)) / (
        log_term * (shift + 2 * x7 * x3 / (log_term * x1**2 * x8) + x3 / x5)
    )


def borehole_high(x: Tensor) -> Tensor:
    return _borehole(x, 2 * math.pi, 1.0)


def borehole_low1(x: Tensor) -> Tensor:
    return _borehole(x, 5.0, 1.5)


def borehole_low2(x: Tensor) -> Tensor:
    return _borehole(x, 7.0, 0.5)


def borehole(num_fidelities: int = 2) -> MultiFidelityFunction:
    levels = {2: (borehole_low1, borehole_high), 3: (borehole_low2, borehole_low1, borehole_high)}
    return MultiFidelityFunction(
        f"borehole_{num_fidelities}f",
        8,
        BOREHOLE_LOWER,
        BOREHOLE_UPPER,
        levels[num_fidelities],
        SOURCE,
    )


# --------------------------------------------------------------- Welded beam (4d, 4 levels)
# Materials: steel, cast iron, aluminum, brass (manuscript table).
WELD_C1 = (0.1047, 0.0489, 0.5235, 0.5584)
WELD_C2 = (0.0481, 0.0224, 0.2405, 0.2566)
WELD_SIGMA_D = (30e3, 8e3, 5e3, 8e3)
WELD_E = (30e6, 14e6, 10e6, 16e6)
WELD_G = (12e6, 6e6, 4e6, 6e6)
WELD_L, WELD_DELTA_MAX, WELD_F = 14.0, 0.25, 6000.0
WELD_LOWER = (0.0625, 0.1, 2.0, 0.0625)  # h, l, t, b
WELD_UPPER = (2.0, 10.0, 20.0, 2.0)


def weld_cost(h, l, t, b, m: int, w: int = 0) -> Tensor:
    """``f = (1 + C1) (w t + l) h^2 + C2 t b (L + l)`` for material index ``m`` in 0..3."""
    return (1 + WELD_C1[m]) * (w * t + l) * h**2 + WELD_C2[m] * t * b * (WELD_L + l)


def weld_constraints(x: Tensor, m: int = 0, w: int = 0) -> Tensor:
    """The five inequality constraints ``g_1..g_5 >= 0``, shape ``(n, 5)``, exactly as printed.

    Caveat: the manuscript's ``J`` (and the ``P_c`` prefactor) differ from the classic Ragsdell &
    Phillips welded beam (``J`` there carries an extra factor 2), so ``g_1`` and ``g_4`` do not
    reproduce the classic optimum (cost 1.7245). ``g_2``, ``g_3``, ``g_5`` and the cost do."""
    h, l, t, b = x.unbind(1)
    s2, F, L = math.sqrt(2.0), WELD_F, WELD_L
    E, G = WELD_E[m], WELD_G[m]
    if w == 0:
        area = s2 * h * l
        j = s2 * h * l * ((h + t) ** 2 / 4 + l**2 / 12)
        r = 0.5 * torch.sqrt(l**2 + (h + t) ** 2)
    else:
        area = s2 * h * (t + l)
        j = s2 * h * l * ((h + t) ** 2 / 4 + l**2 / 12) + s2 * h * t * ((h + l) ** 2 / 4 + t**2 / 12)
        r = torch.maximum(
            0.5 * torch.sqrt(l**2 + (h + t) ** 2), 0.5 * torch.sqrt(t**2 + (h + l) ** 2)
        )
    cos_theta = l / (2 * r)
    tau1 = F / area
    tau2 = F * (L + 0.5 * l) * r / j
    tau = torch.sqrt(tau1**2 + tau2**2 + 2 * tau1 * tau2 * cos_theta)
    sigma = 6 * F * L / (t**2 * b)
    delta = 4 * F * L**3 / (E * t**3 * b)
    pc = 4.013 * t * b**3 * math.sqrt(E * G) / (6 * L**2) * (1 - t / (4 * L) * math.sqrt(E / G))
    sd = WELD_SIGMA_D[m]
    return torch.stack([0.577 * sd - tau, sd - sigma, b - h, pc - F, WELD_DELTA_MAX - delta], 1)


def _weld_level(m: int):
    return lambda x: weld_cost(x[:, 0], x[:, 1], x[:, 2], x[:, 3], m)


def welded_beam() -> MultiFidelityFunction:
    """``f_H`` = steel; the three low fidelities are cast iron, aluminum, brass (``w = 0``)."""
    return MultiFidelityFunction(
        "welded_beam_4f",
        4,
        WELD_LOWER,
        WELD_UPPER,
        (_weld_level(3), _weld_level(2), _weld_level(1), _weld_level(0)),
        SOURCE,
    )


def all_multi_fidelity() -> list[MultiFidelityFunction]:
    return [forrester(), currin(), borehole(2), borehole(3), welded_beam()]
