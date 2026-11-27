"""Fighter airframe lift and drag."""

from math import radians

import numpy as np
from foam.case import at_least, variable
from foam.shapes import prism, revolve, wing
from foam.snappy import surface

from ..external import _DRAG, _case, _turn
from ..external_flow import wind_tunnel


def spitfire(case, design, level):
    """A simplified single-engine fighter airframe with an elliptical wing, half model."""
    angle = radians(design["angle_of_attack_deg"])
    washout = design["wing_twist_deg"]
    fuselage = revolve(
        [0.0, 0.5, 1.5, 3.0, 5.0, 7.0, 9.1], [0.05, 0.42, 0.6, 0.62, 0.45, 0.25, 0.04]
    )
    stations = []
    for y in np.linspace(-0.2, 5.6, 14):
        fraction = max(y, 0.0) / 5.6
        chord = 2.4 * np.sqrt(max(1 - fraction**2, 0.0)) + 0.25
        stations.append((y, 3.6 - 0.45 * chord, -0.25 + 0.08 * fraction * 5.6,
                         chord, washout * fraction))  # fmt: skip
    main = wing(stations, thickness=0.13, camber=0.02)
    tail = wing(
        [(-0.1, 7.8, 0.1, 1.2, 0.0), (1.2, 8.0, 0.1, 0.9, 0.0), (1.7, 8.3, 0.1, 0.4, 0.0)],
        thickness=0.1, camber=0.0,
    )  # fmt: skip
    fin = prism([(7.7, 0.1), (9.0, 0.1), (9.0, 1.4), (8.5, 1.4)], -0.1, 0.05)
    parts = {"fuselage": fuselage, "wing": main, "tailplane": tail, "fin": fin}
    surfaces = {
        name: surface(_turn(t, 1, angle, (3.6, 0.0, 0.0)), (4, 4) if name != "fuselage" else (3, 4))
        for name, t in parts.items()
    }
    return wind_tunnel(
        case, level, surfaces, (-12.0, 0.0, -10.0), (30.0, 14.0, 10.0), 1.0,
        speed=60.0, length=9.1, area=22.5, half=True, ground=False,
        refine=[((-1.0, 0.0, -1.5), (13.0, 6.5, 2.0), 2)], inside=(-7.013, 9.027, 5.031),
    )  # fmt: skip


CASE = _case(
    "spitfire",
    "Fighter airframe lift and drag",
    spitfire,
    (
        variable("angle_of_attack_deg", 1.0, 10.0, 4.0),
        variable("wing_twist_deg", -4.0, 0.0, -2.0),
    ),
    "lift_to_drag",
    (at_least("lift_coefficient", 0.2),),
    maximize=True,
    positive=_DRAG,
    simplification="simplified airframe, no propeller",
)
