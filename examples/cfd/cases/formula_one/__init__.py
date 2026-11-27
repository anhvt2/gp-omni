"""Simplified open-wheel racing car."""

import numpy as np
from foam.case import at_most, variable
from foam.shapes import box, prism, wing
from foam.snappy import surface

from ..external import _DRAG, _case, _wheel
from ..external_flow import wind_tunnel


def formula_one(case, design, level):
    """A simplified open-wheel racing car, half model."""
    ride = design["ride_height_m"]
    front = -design["front_wing_deg"]
    rear = -design["rear_wing_deg"]
    nose = prism(
        [(0.5, ride + 0.2), (1.8, ride + 0.05), (4.4, ride + 0.05), (4.4, ride + 0.5),
         (3.0, ride + 0.75), (1.8, ride + 0.5)], -0.05, 0.35,
    )  # fmt: skip
    floor = box((1.6, -0.05, ride), (4.3, 0.75, ride + 0.07))
    front_wing = wing(
        [(-0.05, 0.0, ride + 0.07, 0.55, front), (0.95, 0.0, ride + 0.07, 0.55, front)],
        thickness=0.12, camber=-0.07,
    )  # fmt: skip
    rear_wing = wing(
        [(-0.05, 4.55, 0.85, 0.45, rear), (0.5, 4.55, 0.85, 0.45, rear)],
        thickness=0.12, camber=-0.09,
    )  # fmt: skip
    pylon = box((4.55, -0.05, ride + 0.4), (4.75, 0.06, 0.87))
    wheels = np.concatenate([_wheel(1.0, 0.65, 0.95, 0.33), _wheel(4.0, 0.7, 1.05, 0.34)])
    axles = np.concatenate(
        [box((0.95, 0.0, 0.28), (1.05, 0.7, 0.36)), box((3.95, 0.0, 0.28), (4.05, 0.75, 0.36))]
    )
    parts = {
        "body": nose, "floor": floor, "front_wing": front_wing, "rear_wing": rear_wing,
        "pylon": pylon, "wheels": wheels, "axles": axles,
    }  # fmt: skip
    return wind_tunnel(
        case, level,
        {n: surface(t, (4, 4) if "wing" in n else (3, 4)) for n, t in parts.items()},
        (-6.0, 0.0, 0.0), (18.0, 5.0, 4.0), 0.4, speed=60.0, length=5.0, area=1.4, half=True,
        refine=[((-0.8, 0.0, 0.0), (7.5, 1.6, 1.5), 2)],
    )  # fmt: skip


CASE = _case(
    "formula_one",
    "Simplified open-wheel racing car",
    formula_one,
    (
        variable("front_wing_deg", 2.0, 20.0, 10.0),
        variable("rear_wing_deg", 5.0, 30.0, 16.0),
        variable("ride_height_m", 0.05, 0.12, 0.08),
    ),
    "downforce_coefficient",
    (at_most("drag_coefficient", 0.15),),
    maximize=True,
    positive=_DRAG,
    simplification="simplified body, static wheels and ground",
)
