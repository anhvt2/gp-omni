"""Axisymmetric poppet valve."""

from math import pi, radians, tan

import numpy as np
from foam import measure as post
from foam.boundaries import flow_inlet, pressure_outlet, slip, wall, wedge
from foam.case import at_most, run_plan, variable
from foam.hex_mesh import hex_mesh
from foam.physics import flow, write_flow_case

from ..internal import WATER, WEDGES, _case, _finish, _wedge_grid


def hydraulic_valve(case, design, level):
    pipe_radius, stem, outlet_radius = 0.01, 0.0085, 0.011
    seat, lift = design["seat_radius_m"], design["lift_m"]
    slope = tan(radians(design["cone_half_angle_deg"]))
    seat_x = 0.03
    tip = seat_x + lift - seat / slope  # the cone leaves a gap of lift*slope at the seat
    shoulder = tip + stem / slope
    x = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, max(tip - 0.002, 0.005), level.cells(30)),
                np.linspace(max(tip - 0.002, 0.005), shoulder + 0.006, level.cells(90)),
                np.linspace(shoulder + 0.006, 0.11, level.cells(50)),
            ]
        )
    )
    outer = np.interp(
        x,
        [0.0, seat_x - 0.008, seat_x - 0.004, seat_x, seat_x + 0.002, 0.11],
        [pipe_radius, pipe_radius, seat, seat, outlet_radius, outlet_radius],
    )
    inner = np.clip((x - tip) * slope, 0.0, stem)
    inner = np.where(x > 0.07, np.clip(stem * (0.085 - x) / 0.015, 0.0, stem), inner)
    if np.min(outer - inner) <= 2e-4:
        raise ValueError("the poppet closes the seat")
    mesh = hex_mesh()
    grid = _wedge_grid(x, inner, outer, level.cells(30))
    middle = 0.5 * (x[1:] + x[:-1])
    core = np.where((middle > tip) & (middle < 0.085), "plug", "axis").astype(object)
    mesh.add_block(
        grid,
        {
            "i0": "inlet", "i1": "outlet", "j0": core[:, None], "j1": "walls",
            "k0": "wedge_back", "k1": "wedge_front",
        },
    )  # fmt: skip
    rate = 2.0e-4
    patches = _finish(
        case,
        mesh,
        {
            "inlet": flow_inlet(rate / WEDGES), "outlet": pressure_outlet(), "walls": wall(),
            "plug": wall(), "axis": slip(), "wedge_back": wedge(), "wedge_front": wedge(),
        },
    )  # fmt: skip
    functions = post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet")
    functions |= post.forces("plug", ["plug"], WATER["density"])
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam",
            velocity=rate / (pi * pipe_radius**2),
            length=0.004,
            iterations=level.iterations,
            second_order=level.resolution > 1,
            functions=functions,
            **WATER,
        ),
    )

    def measure(directory):
        loss = post.last_surface(directory, "total_p_inlet") - post.last_surface(
            directory, "total_p_outlet"
        )
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "pressure_loss_pa": WATER["density"] * loss,
            "flow_force_n": abs(float(post.last_force(directory, "plug")[0])) * WEDGES,
        }

    return run_plan(solve=[["simpleFoam"]], measure=measure)


CASE = _case(
    "hydraulic_valve",
    "Axisymmetric poppet valve",
    hydraulic_valve,
    (
        variable("cone_half_angle_deg", 30.0, 55.0, 42.0),
        variable("lift_m", 0.0012, 0.003, 0.002),
        variable("seat_radius_m", 0.0055, 0.0072, 0.0064),
    ),
    "pressure_loss_pa",
    (at_most("flow_force_n", 0.15),),
    positive=("pressure_loss_pa",),
    simplification="axisymmetric, no structural analysis",
)
