"""Bend with nine free diameters."""

from math import pi

import numpy as np
from foam.case import at_most, variable
from foam.ducts import u_bend

from ..internal import _bend, _case

STATIONS = tuple(f"diameter_{n}_m" for n in range(1, 10))


def parametric_pipe(case, design, level):
    radius = 0.25
    _, total = u_bend(0.4, radius, 0.8)
    # nine stations spread evenly through the bend, between fixed 100 mm ends
    along = (0.4 + np.linspace(0.05, 0.95, 9) * pi * radius) / total
    knots = np.concatenate([[0.0, 0.4 / total], along, [(0.4 + pi * radius) / total, 1.0]])
    values = np.array([0.1, 0.1, *(design[n] for n in STATIONS), 0.1, 0.1]) / 2
    return _bend(case, level, radius, lambda s: np.interp(s, knots, values), 0.01, 0.1)


CASE = _case(
    "parametric_pipe",
    "Bend with nine free diameters",
    parametric_pipe,
    tuple(variable(n, 0.085, 0.12, 0.1) for n in STATIONS),
    "pressure_loss_pa",
    (at_most("volume_m3", 0.05),),
    positive=("pressure_loss_pa",),
)
