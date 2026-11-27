"""Water pipe with a 180 degree bend."""

import numpy as np
from foam.case import at_most, variable

from ..internal import _bend, _case


def pipe(case, design, level):
    diameter = design["diameter_m"]
    return _bend(
        case,
        level,
        design["bend_radius_m"],
        lambda s: np.full(len(s), diameter / 2),
        0.01,
        diameter,
    )


CASE = _case(
    "pipe",
    "Water pipe with a 180 degree bend",
    pipe,
    (variable("diameter_m", 0.08, 0.12, 0.1), variable("bend_radius_m", 0.15, 0.4, 0.25)),
    "pressure_loss_pa",
    (at_most("volume_m3"),),
    positive=("pressure_loss_pa",),
    simplification="flow only, no structural analysis",
)
