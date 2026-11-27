"""Tall building in an atmospheric wind profile."""

from math import cos, radians, sin

import numpy as np
from foam.boundaries import wind_profile_inlet
from foam.case import at_least, variable
from foam.shapes import loft, volume
from foam.snappy import surface

from ..external import _case
from ..external_flow import wind_tunnel


def building_wind_load(case, design, level):
    """A 1:200 tower, tapered, twisted and with chamfered corners, in a wind profile."""
    height, half = 1.0, 0.13
    taper, twist = design["taper_ratio"], radians(design["twist_deg"])
    cut = design["chamfer_fraction"] * 2 * half
    corner = np.array(
        [
            (half, -half + cut), (half, half - cut), (half - cut, half), (-half + cut, half),
            (-half, half - cut), (-half, -half + cut), (-half + cut, -half),
            (half - cut, -half),
        ]
    )  # fmt: skip
    sections = []
    for z in np.linspace(0.0, height, 25):
        fraction = z / height
        scale = 1 - (1 - taper) * fraction
        c, s = cos(twist * fraction), sin(twist * fraction)
        xy = scale * corner @ np.array([[c, s], [-s, c]])
        sections.append(np.column_stack([xy, np.full(len(corner), z)]))
    tower = loft(sections)
    return wind_tunnel(
        case, level, {"tower": surface(tower, (3, 3))},
        (-2.5, -2.0, 0.0), (6.0, 2.0, 2.6), 0.18, speed=10.0, length=height,
        area=2 * half * height, refine=[((-0.5, -0.6, 0.0), (1.8, 0.6, 1.3), 2)],
        overrides={"inlet": wind_profile_inlet(10.0, height, 0.005)},
        libraries=("atmosphericModels",), extra={"volume_m3": volume(tower)},
    )  # fmt: skip


CASE = _case(
    "building_wind_load",
    "Tall building in an atmospheric wind profile",
    building_wind_load,
    (
        variable("taper_ratio", 0.6, 1.0, 0.85),
        variable("twist_deg", 0.0, 90.0, 30.0),
        variable("chamfer_fraction", 0.02, 0.2, 0.08),
    ),
    "base_moment_coefficient",
    (at_least("volume_m3", 0.05),),
    positive=("base_moment_coefficient", "drag_coefficient"),
    simplification="generic tower, steady wind",
)
