"""Train car nose shape and yaw."""

from math import pi, radians

import numpy as np
from foam.case import at_least, variable
from foam.shapes import loft, volume
from foam.snappy import surface

from ..external import _DRAG, _case, _turn
from ..external_flow import wind_tunnel


def _train_body(nose: float, exponent: float, length: float = 5.0) -> list:
    """Cross sections of a train car with equal rounded nose and tail."""
    angle = np.linspace(0, 2 * pi, 40, endpoint=False)
    sections = []
    for x in np.concatenate(
        [np.linspace(0, nose, 14), np.linspace(nose, length - nose, 4)[1:-1],
         np.linspace(length - nose, length, 14)]
    ):  # fmt: skip
        reach = min(x, length - x, nose) / nose
        scale = max(reach**exponent, 0.04)
        # a rounded rectangle 0.5 m wide and 0.6 m tall
        shape_y = np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** 0.4
        shape_z = np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** 0.4
        sections.append(
            np.column_stack(
                [np.full(40, x), 0.25 * scale * shape_y, 0.1 + 0.3 * scale * (shape_z + 1)]
            )
        )
    return sections


def train_aerodynamics(case, design, level):
    body = loft(_train_body(design["nose_length_m"], design["nose_exponent"]))
    yawed = _turn(body, 2, radians(design["yaw_deg"]), (2.5, 0.0, 0.0))
    return wind_tunnel(
        case, level, {"train": surface(yawed, (3, 3))},
        (-3.0, -2.0, 0.0), (12.0, 2.5, 2.4), 0.22, speed=45.0, length=5.0, area=0.5 * 0.6,
        refine=[((-0.5, -0.8, 0.0), (7.0, 1.0, 1.0), 1)],
        extra={"volume_m3": volume(body)},
    )  # fmt: skip


CASE = _case(
    "train_aerodynamics",
    "Train car nose shape and yaw",
    train_aerodynamics,
    (
        variable("nose_length_m", 0.5, 1.4, 0.9),
        variable("nose_exponent", 0.35, 0.8, 0.55),
        variable("yaw_deg", 0.0, 10.0, 3.0),
    ),
    "drag_coefficient",
    (at_least("volume_m3", 0.05),),
    positive=_DRAG,
    simplification="single car, stationary ground",
)
