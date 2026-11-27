"""Spinning dimpled ball."""

from math import pi

import numpy as np
from foam.boundaries import rotating_wall
from foam.case import at_least, variable
from foam.shapes import sphere
from foam.snappy import surface

from ..external import _DRAG, _case
from ..external_flow import wind_tunnel

BALL_RADIUS = 0.02134


def _dimples(depth: float, count: int = 90):
    """Smooth spherical caps spread on a Fibonacci lattice."""
    n = np.arange(count) + 0.5
    z = 1 - 2 * n / count
    azimuth = pi * (1 + 5**0.5) * n
    ring = np.sqrt(1 - z**2)
    centres = np.column_stack([ring * np.cos(azimuth), ring * np.sin(azimuth), z])
    half_angle = 0.14

    def relief(direction: np.ndarray) -> np.ndarray:
        nearest = np.arccos(np.clip(direction @ centres.T, -1, 1)).min(axis=1)
        return depth * np.clip(1 - (nearest / half_angle) ** 2, 0, None)

    return relief


def dimpled_ball(case, design, level):
    spin = design["spin_rad_s"]
    ball = sphere(BALL_RADIUS, segments=96, relief=_dimples(design["dimple_depth_m"]))
    return wind_tunnel(
        case, level, {"ball": surface(ball, (4, 4))},
        (-0.2, -0.18, -0.18), (0.55, 0.18, 0.18), 0.024, speed=40.0, length=2 * BALL_RADIUS,
        area=pi * BALL_RADIUS**2, ground=False,
        refine=[((-0.04, -0.04, -0.04), (0.14, 0.04, 0.04), 3)],
        overrides={"ball": rotating_wall(spin, axis=(0, 1, 0))},
        time_step=2e-5, compressible=True, inside=(-0.1513, 0.1117, 0.1231),
    )  # fmt: skip


CASE = _case(
    "dimpled_ball",
    "Spinning dimpled ball",
    dimpled_ball,
    (
        variable("dimple_depth_m", 0.0002, 0.001, 0.0005),
        variable("spin_rad_s", 200.0, 700.0, 628.0),
    ),
    "drag_coefficient",
    (at_least("lift_coefficient", 0.25),),
    solver="rhoPimpleFoam",
    positive=_DRAG,
    simplification="short transient, spinning surface without moving dimples, no acoustics",
)
