"""Ahmed reference car body, half model."""

from math import radians, tan

import numpy as np
from foam.case import at_least, variable
from foam.shapes import prism, volume
from foam.snappy import surface

from ..external import _DRAG, _case
from ..external_flow import wind_tunnel


def _ahmed_outline(slant_deg: float, slant: float, diffuser_deg: float, nose: float):
    """Convex side view of the Ahmed body, 1.044 m long on a 0.05 m ground clearance."""
    length, low, high, diffuser = 1.044, 0.05, 0.338, 0.2
    angle = np.linspace(0, np.pi / 2, 9)
    lower_nose = [(nose * (1 - np.cos(a)), low + nose * (1 - np.sin(a))) for a in angle]
    upper_nose = [(nose * (1 - np.sin(a)), high - nose * (1 - np.cos(a))) for a in angle]
    tail = [
        (length - diffuser, low),
        (length, low + diffuser * tan(radians(diffuser_deg))),
        (length, high - slant * np.sin(radians(slant_deg))),
        (length - slant * np.cos(radians(slant_deg)), high),
    ]
    return [*lower_nose, *tail, *upper_nose]


def ahmed_body(case, design, level):
    outline = _ahmed_outline(
        design["slant_angle_deg"],
        design["slant_length_m"],
        design["diffuser_angle_deg"],
        design["nose_radius_m"],
    )
    half_width = 0.1945
    body = prism(outline, -0.04, half_width)
    return wind_tunnel(
        case,
        level,
        {"body": surface(body, (3, 4))},
        (-2.0, 0.0, 0.0),
        (5.0, 1.5, 1.5),
        0.16,
        speed=40.0,
        length=1.044,
        area=0.389 * 0.288,
        half=True,
        refine=[((-0.4, 0.0, 0.0), (2.2, 0.5, 0.6), 2)],
        extra={"volume_m3": abs(volume(prism(outline, 0.0, half_width))) * 2},
    )


CASE = _case(
    "ahmed_body",
    "Ahmed reference car body, half model",
    ahmed_body,
    (
        variable("slant_angle_deg", 12.0, 35.0, 25.0),
        variable("slant_length_m", 0.15, 0.25, 0.222),
        variable("diffuser_angle_deg", 0.0, 8.0, 2.0),
        variable("nose_radius_m", 0.06, 0.12, 0.1),
    ),
    "drag_coefficient",
    (at_least("volume_m3", 0.03),),
    positive=_DRAG,
    simplification="no support stilts, rounding only in side view",
)
