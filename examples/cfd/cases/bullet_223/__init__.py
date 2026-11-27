"""Supersonic bullet."""

from math import pi, radians, tan

import numpy as np
from foam import measure as post
from foam.boundaries import open_outlet, slip, supersonic_inlet, wall, wedge
from foam.case import at_least, run_plan, variable
from foam.hex_mesh import hex_mesh
from foam.physics import flow, write_flow_case

from ..external_structured import _case, _graded
from ..internal import WEDGES, _finish, _wedge_grid


def bullet_223(case, design, level):
    """A 23 mm bullet at 900 m/s, axisymmetric, with a short closing tail."""
    radius, length = 0.00285, 0.023
    nose, tail = design["nose_length_m"], design["boat_tail_length_m"]
    base = radius - tail * tan(radians(design["boat_tail_angle_deg"]))
    closing = 1.5 * base
    end = length + closing
    x = np.concatenate(
        [
            np.linspace(-0.012, 0.0, level.cells(30))[:-1],
            np.linspace(0.0, end, level.cells(150))[:-1],
            end + _graded(end / 150, 0.05, level.cells(50)),
        ]
    )
    along = np.clip(x / nose, 0, 1)
    # tangent ogive, cylinder, boat tail and closing cone
    rho = (radius**2 + nose**2) / (2 * radius)
    ogive = np.sqrt(np.maximum(rho**2 - (nose * (1 - along)) ** 2, 0)) + radius - rho
    body = np.interp(x, [nose, length - tail, length, end], [radius, radius, base, 0.0], right=0.0)
    body = np.where(x < nose, ogive, body)
    body = np.where((x <= 0) | (x >= end), 0.0, body)
    outer = np.full(len(x), 0.025)
    mesh = hex_mesh()
    surface = np.where((x[1:] + x[:-1] > 0) & (x[1:] + x[:-1] < 2 * end), "body", "axis")
    mesh.add_block(
        _wedge_grid(x, body, outer, level.cells(50)),
        {
            "i0": "inlet", "i1": "outlet", "j0": surface.astype(object)[:, None], "j1": "far",
            "k0": "wedge_back", "k1": "wedge_front",
        },
    )  # fmt: skip
    speed, pressure, temperature = 900.0, 101325.0, 288.0
    patches = _finish(
        case,
        mesh,
        {
            "inlet": supersonic_inlet((speed, 0, 0), pressure, temperature),
            "outlet": open_outlet(), "far": open_outlet(), "body": wall(), "axis": slip(),
            "wedge_back": wedge(), "wedge_front": wedge(),
        },
    )  # fmt: skip
    functions = post.forces("body", ["body"], None, reference_pressure=pressure)
    functions |= post.flux("inlet", "outlet")
    model = flow(
        "rhoCentralFoam", velocity=speed, length=radius, turbulent=False,
        time_step=4e-8, time_steps=level.time_steps, pressure=pressure,
        temperature=temperature, functions=functions,
    )  # fmt: skip
    write_flow_case(case, patches, model)
    dense = np.linspace(0, end, 2000)
    profile = np.interp(dense, x, body)
    mass = float(np.trapezoid(pi * profile**2, dense)) * 10500.0

    def measure(directory):
        drag = float(post.last_force(directory, "body")[0]) * WEDGES
        dynamic = 0.5 * pressure / (287.0 * temperature) * speed**2 * pi * radius**2
        return {
            "flow_mismatch": post.imbalance(directory, "inlet", "outlet"),
            "drag_coefficient": drag / dynamic,
            "mass_g": 1000.0 * mass,
        }

    return run_plan(solve=[["rhoCentralFoam"]], measure=measure)


CASE = _case(
    "bullet_223",
    "Supersonic bullet",
    bullet_223,
    (
        variable("nose_length_m", 0.008, 0.013, 0.0105),
        variable("boat_tail_length_m", 0.0015, 0.0045, 0.003),
        variable("boat_tail_angle_deg", 4.0, 12.0, 8.0),
    ),
    "drag_coefficient",
    (at_least("mass_g"),),
    solver="rhoCentralFoam",
    positive=("drag_coefficient",),
    simplification="axisymmetric short transient, no structural analysis",
)
