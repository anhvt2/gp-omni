"""Butterfly valve in a water line."""

from math import cos, pi, radians, sin

import numpy as np
from foam import measure as post
from foam.boundaries import flow_inlet, pressure_outlet, wall
from foam.cad import fluid_surfaces, tube
from foam.case import at_most, run_plan, variable

from ..internal_cad import WATER, _case, _snappy_fluid, _steady, _total


def water_valve(case, design, level):
    """A butterfly valve in a 50 mm water line, at 1 litre per second."""
    radius = 0.025
    opening = radians(design["opening_deg"])
    thickness, seat = design["disc_thickness_m"], design["seat_height_m"]
    disc_radius = radius - seat - 0.001

    def build(occ):
        line = [tube(occ, (-0.12, 0, 0), (0.2, 0, 0), radius)]
        # the disc turns about the y axis; at 90 degrees it lies along the flow
        axis = np.array([sin(radians(90) - opening), 0.0, cos(radians(90) - opening)])
        normal = np.array([axis[2], 0.0, -axis[0]])
        start = -0.5 * thickness * normal
        disc = tube(occ, start, start + thickness * normal, disc_radius)
        ring = tube(occ, (-0.034, 0, 0), (-0.026, 0, 0), radius)
        bore = tube(occ, (-0.034, 0, 0), (-0.026, 0, 0), radius - seat)
        seat_ring, _ = occ.cut([ring], [bore])
        fluid, _ = occ.cut(line, [disc, *seat_ring])
        return fluid

    def classify(centre):
        if abs(centre[0] + 0.12) < 1e-6:
            return "inlet"
        if abs(centre[0] - 0.2) < 1e-6:
            return "outlet"
        if np.linalg.norm(centre) < 0.0045:
            return "disc"
        return "walls"

    surfaces, _ = fluid_surfaces(build, classify, 0.004)
    if "disc" not in surfaces:
        raise ValueError("the disc surface was not found")
    rate = 1.0e-3
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {"inlet": flow_inlet(rate), "outlet": pressure_outlet(), "walls": wall(), "disc": wall()},
        0.006, (-0.0871, 0.0013, 0.0017),
    )  # fmt: skip
    functions = post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet")
    functions |= post.forces("disc", ["disc"], WATER["density"])
    _steady(case, patches, rate / (pi * radius**2), 2 * radius, level, functions, **WATER)

    def measure(directory):
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "pressure_loss_pa": WATER["density"]
            * (_total(directory, "inlet") - _total(directory, "outlet")),
            "flow_force_n": float(np.linalg.norm(post.last_force(directory, "disc"))),
        }

    return run_plan(mesh=commands, solve=[["simpleFoam"]], measure=measure)


CASE = _case(
    "water_valve",
    "Butterfly valve in a water line",
    water_valve,
    (
        variable("opening_deg", 35.0, 80.0, 55.0),
        variable("disc_thickness_m", 0.003, 0.007, 0.005),
        variable("seat_height_m", 0.0012, 0.004, 0.0025),
    ),
    "pressure_loss_pa",
    (at_most("flow_force_n", 0.2),),
    positive=("pressure_loss_pa",),
)
