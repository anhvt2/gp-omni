"""Simplified notchback sedan."""

from math import radians, tan

import numpy as np
from foam.case import at_least, variable
from foam.shapes import box, prism, volume
from foam.snappy import surface

from ..external import _DRAG, _case, _wheel
from ..external_flow import wind_tunnel


def drivaer(case, design, level):
    """An own simplified notchback sedan inspired by the DrivAer shape family, half model."""
    slant = radians(design["rear_slant_deg"])
    diffuser = radians(design["diffuser_angle_deg"])
    deck = design["trunk_height_m"]
    floor, tail = 0.16, 4.6
    lower = [
        (0.0, 0.42), (0.18, 0.22), (0.7, floor), (3.8, floor),
        (tail, floor + 0.8 * tan(diffuser)), (tail, deck), (3.9, deck + 0.03),
        (1.15, 0.92), (0.15, 0.68),
    ]  # fmt: skip
    roof, roof_start, roof_end = 1.42, 2.05, 2.85
    rear_foot = min(roof_end + (roof - deck + 0.05) / tan(slant), tail - 0.1)
    cabin = [
        (1.05, 0.85), (rear_foot, 0.85), (rear_foot, deck - 0.02), (roof_end, roof),
        (roof_start, roof),
    ]  # fmt: skip
    wheels = np.concatenate([_wheel(0.95, 0.62, 0.92, 0.32), _wheel(3.75, 0.62, 0.92, 0.32)])
    axles = np.concatenate(
        [box((0.9, 0.0, 0.27), (1.0, 0.68, 0.34)), box((3.7, 0.0, 0.27), (3.8, 0.68, 0.34))]
    )
    shell, top = prism(lower, -0.05, 0.82), prism(cabin, -0.05, 0.7)
    parts = {"body": shell, "cabin": top, "wheels": wheels, "axles": axles}
    size = abs(volume(prism(lower, 0.0, 0.82))) * 2 + abs(volume(prism(cabin, 0.0, 0.7))) * 2
    return wind_tunnel(
        case, level, {n: surface(t, (3, 4)) for n, t in parts.items()},
        (-7.0, 0.0, 0.0), (18.0, 5.0, 4.5), 0.4, speed=40.0, length=4.6, area=2.16, half=True,
        refine=[((-0.8, 0.0, 0.0), (7.5, 1.5, 1.9), 2)], extra={"volume_m3": size},
    )  # fmt: skip


CASE = _case(
    "drivaer",
    "Simplified notchback sedan",
    drivaer,
    (
        variable("rear_slant_deg", 18.0, 35.0, 26.0),
        variable("diffuser_angle_deg", 0.0, 8.0, 3.0),
        variable("trunk_height_m", 0.9, 1.15, 1.02),
    ),
    "drag_coefficient",
    (at_least("volume_m3", 0.03),),
    positive=_DRAG,
    simplification="own geometry, not the published DrivAer model; static wheels and ground",
)
