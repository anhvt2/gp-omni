"""Idealized upper airway."""

from math import cos, pi, radians, sin

import numpy as np
from foam import measure as post
from foam.boundaries import flow_inlet, pressure_outlet, wall
from foam.cad import fluid_surfaces, fuse, name_caps, tube
from foam.case import at_least, run_plan, variable

from ..internal_cad import AIR_DENSITY, _case, _snappy_fluid, _steady, _total


def human_upper_airways(case, design, level):
    """An idealized trachea with a narrowing, dividing into two main bronchi."""
    radius = 0.009
    narrow = radius * design["constriction_ratio"]
    position = design["constriction_position"] * 0.1
    angle = radians(design["branch_angle_deg"])
    branch, length = 0.0065, 0.06
    left = (-length * sin(angle), 0.0, -0.12 - length * cos(angle))
    right = (length * sin(angle), 0.0, -0.12 - length * cos(angle))

    def build(occ):
        parts = [
            tube(occ, (0, 0, 0), (0, 0, -position + 0.012), radius),
            tube(occ, (0, 0, -position + 0.012), (0, 0, -position), radius, narrow),
            tube(occ, (0, 0, -position), (0, 0, -position - 0.012), narrow, radius),
            tube(occ, (0, 0, -position - 0.012), (0, 0, -0.12), radius),
            (3, occ.addSphere(0, 0, -0.12, radius)),
            tube(occ, (0, 0, -0.12), left, branch),
            tube(occ, (0, 0, -0.12), right, branch),
        ]
        return fuse(occ, parts)

    caps = {"mouth": (0, 0, 0), "left": left, "right": right}
    surfaces, _ = fluid_surfaces(build, name_caps(caps), 0.002)
    rate = 5.0e-4  # 30 litres per minute
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {"mouth": flow_inlet(rate), "left": pressure_outlet(), "right": pressure_outlet(),
         "walls": wall()},
        0.0022, (0.0007, 0.0011, -0.0053), (1, 1),
    )  # fmt: skip
    names = ("mouth", "left", "right")
    functions = post.flux(*names) | post.total_pressure(*names)
    _steady(case, patches, rate / (pi * radius**2), 2 * radius, level, functions)

    def measure(directory):
        flows = {n: abs(post.last_flux(directory, n)) for n in names}
        out = np.average(
            [_total(directory, n) for n in names[1:]], weights=[flows["left"], flows["right"]]
        )
        return {
            "flow_imbalance": post.imbalance(directory, *names),
            "pressure_drop_pa": AIR_DENSITY * float(_total(directory, "mouth") - out),
            "left_flow_fraction": flows["left"] / (flows["left"] + flows["right"]),
        }

    return run_plan(mesh=commands, solve=[["simpleFoam"]], measure=measure)


CASE = _case(
    "human_upper_airways",
    "Idealized upper airway",
    human_upper_airways,
    (
        variable("constriction_ratio", 0.55, 0.9, 0.7),
        variable("constriction_position", 0.25, 0.7, 0.45),
        variable("branch_angle_deg", 25.0, 55.0, 38.0),
    ),
    "pressure_drop_pa",
    (at_least("left_flow_fraction", 0.15),),
    positive=("pressure_drop_pa",),
    simplification="idealized tubes, steady inhalation",
)
