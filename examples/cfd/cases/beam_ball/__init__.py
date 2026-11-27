"""Ball on a cantilever rod."""

from math import pi

import numpy as np
from foam import measure as post
from foam.case import at_most, variable
from foam.shapes import loft, sphere
from foam.snappy import surface

from ..external import _case
from ..external_flow import DENSITY, wind_tunnel


def beam_ball(case, design, level):
    """A sphere on a cantilever rod in a stream; deflection from beam theory."""
    radius, length = design["ball_radius_m"], design["beam_length_m"]
    rod_radius, modulus = 0.004, 70.0e9
    ball = sphere(radius, centre=(0.0, 0.0, length), segments=48)
    angle = np.linspace(0, 2 * pi, 20, endpoint=False)
    rod = loft(
        [
            np.column_stack(
                [rod_radius * np.cos(angle), rod_radius * np.sin(angle), np.full(20, z)]
            )
            for z in (-0.01, length)
        ]
    )
    speed = 30.0
    functions = post.forces("ball", ["ball"], DENSITY) | post.forces("rod", ["rod"], DENSITY)

    def deflection(directory, metrics):
        tip = float(post.last_force(directory, "ball")[0])
        spread = float(post.last_force(directory, "rod")[0])
        stiffness = modulus * pi * rod_radius**4 / 4
        bend = tip * length**3 / (3 * stiffness) + spread * length**3 / (8 * stiffness)
        return {"deflection_m": bend, "drag_n": tip + spread}

    return wind_tunnel(
        case, level, {"ball": surface(ball, (3, 4)), "rod": surface(rod, (4, 4))},
        (-0.3, -0.25, 0.0), (0.8, 0.25, 0.5), 0.032, speed=speed, length=2 * radius,
        area=pi * radius**2, refine=[((-0.06, -0.06, 0.0), (0.2, 0.06, length + 0.06), 2)],
        functions=functions, derive=deflection, inside=(-0.2013, 0.1517, 0.4131),
    )  # fmt: skip


CASE = _case(
    "beam_ball",
    "Ball on a cantilever rod",
    beam_ball,
    (variable("ball_radius_m", 0.015, 0.035, 0.025), variable("beam_length_m", 0.12, 0.3, 0.2)),
    "drag_n",
    (at_most("deflection_m", 0.2),),
    positive=("drag_n", "deflection_m"),
    simplification="one-way estimate from beam theory, no fluid-structure coupling",
)
