"""Pelton turbine distributor."""

from math import cos, pi, radians, sin

import numpy as np
from foam import measure as post
from foam.boundaries import flow_inlet, pressure_outlet, wall
from foam.cad import fluid_surfaces, fuse, name_caps, tube
from foam.case import at_most, run_plan, variable

from ..internal_cad import WATER, _case, _snappy_fluid, _steady, _total


def pelton_distributor(case, design, level):
    """A tapering main pipe feeding three nozzles of a Pelton turbine."""
    angle = radians(design["branch_angle_deg"])
    branch, end = design["branch_radius_m"], design["end_radius_m"]
    main = 0.1
    stations = (0.37, 0.69, 0.97)
    reach = 0.35
    tips = [(x + reach * cos(angle), reach * sin(angle), 0.0) for x in stations]

    def build(occ):
        parts = [
            tube(occ, (0, 0, 0), (0.28, 0, 0), main),
            tube(occ, (0.28, 0, 0), (1.04, 0, 0), main, end),
        ]
        for start, stop in zip(stations, tips):
            middle = tuple(a + 0.75 * (b - a) for a, b in zip((start, 0, 0), stop))
            parts.append(tube(occ, (start, 0, 0), middle, branch))
            parts.append(tube(occ, middle, stop, branch, 0.6 * branch))
        return fuse(occ, parts)

    outlets = ["nozzle_1", "nozzle_2", "nozzle_3"]
    caps = {"inlet": (0, 0, 0), **dict(zip(outlets, tips))}
    surfaces, _ = fluid_surfaces(build, name_caps(caps), 0.02)
    rate = 0.08
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {"inlet": flow_inlet(rate), **{n: pressure_outlet() for n in outlets}, "walls": wall()},
        0.024, (0.1013, 0.0071, 0.0053), (1, 1),
    )  # fmt: skip
    names = ("inlet", *outlets)
    functions = post.flux(*names) | post.total_pressure(*names)
    _steady(case, patches, rate / (pi * main**2), 2 * main, level, functions, **WATER)

    def measure(directory):
        flows = np.array([abs(post.last_flux(directory, n)) for n in outlets])
        out = np.average([_total(directory, n) for n in outlets], weights=flows)
        return {
            "flow_imbalance": post.imbalance(directory, *names),
            "nozzle_imbalance": float(np.ptp(flows) / flows.mean()),
            "pressure_loss_pa": WATER["density"] * float(_total(directory, "inlet") - out),
        }

    return run_plan(mesh=commands, solve=[["simpleFoam"]], measure=measure)


CASE = _case(
    "pelton_distributor",
    "Pelton turbine distributor",
    pelton_distributor,
    (
        variable("branch_angle_deg", 30.0, 65.0, 45.0),
        variable("branch_radius_m", 0.04, 0.06, 0.05),
        variable("end_radius_m", 0.055, 0.09, 0.07),
    ),
    "nozzle_imbalance",
    (at_most("pressure_loss_pa", 0.2),),
    positive=("pressure_loss_pa",),
    simplification="three nozzles, single phase",
)
