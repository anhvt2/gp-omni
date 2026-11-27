"""Labyrinth seal filling with water."""

import numpy as np
from foam import measure as post
from foam.boundaries import pressure_outlet, total_pressure_inlet, wall, wedge
from foam.case import at_most, run_plan, variable
from foam.hex_mesh import hex_mesh
from foam.physics import flow, write_flow_case

from ..internal import WEDGES, _case, _finish, _wedge_grid


def labyrinth_maze(case, design, level):
    """Water driven into an air-filled annular gap sealed by square teeth."""
    shaft, bore = 0.02, 0.0235
    teeth = round(design["tooth_count"])
    clearance, length = design["clearance_m"], design["length_m"]
    x = np.linspace(0.0, length, level.cells(200) + 1)
    radial = level.cells(18, 10)
    mesh = hex_mesh()
    # teeth rise from the shaft over the middle 80 percent of the length
    phase = np.clip((x - 0.1 * length) / (0.8 * length), 0, 1) * teeth
    ramp = 0.12
    local = np.mod(phase, 1.0)
    shape = np.clip(np.minimum((local - 0.25) / ramp, (0.75 - local) / ramp), 0, 1)
    shape[(x <= 0.1 * length) | (x >= 0.9 * length)] = 0.0
    inner = shaft + (bore - clearance - shaft) * shape
    mesh.add_block(
        _wedge_grid(x, inner, np.full(len(x), bore), radial),
        {
            "i0": "inlet", "i1": "outlet", "j0": "shaft", "j1": "walls",
            "k0": "wedge_back", "k1": "wedge_front",
        },
    )  # fmt: skip
    pressure = 20.0  # m2/s2, so a 20 kPa water head
    patches = _finish(
        case,
        mesh,
        {
            "inlet": total_pressure_inlet(pressure * 1000.0, water=1.0),
            "outlet": pressure_outlet(), "shaft": wall(), "walls": wall(),
            "wedge_back": wedge(), "wedge_front": wedge(),
        },
    )  # fmt: skip
    functions = post.flux("inlet", "outlet")
    write_flow_case(
        case,
        patches,
        flow(
            "interFoam",
            velocity=5.0,
            length=clearance,
            turbulent=False,
            time_step=2e-6,
            time_steps=level.time_steps,
            gravity=(0.0, 0.0, 0.0),
            functions=functions,
        ),
    )

    def measure(directory):
        return {
            "flow_mismatch": post.imbalance(directory, "inlet", "outlet"),
            "leakage_m3_s": abs(post.last_flux(directory, "outlet")) * WEDGES,
            "inflow_m3_s": abs(post.last_flux(directory, "inlet")) * WEDGES,
            "length_m": length,
        }

    return run_plan(solve=[["interFoam"]], measure=measure)


CASE = _case(
    "labyrinth_maze",
    "Labyrinth seal filling with water",
    labyrinth_maze,
    (
        variable("tooth_count", 3.0, 7.0, 5.0),
        variable("clearance_m", 0.0004, 0.0012, 0.0007),
        variable("length_m", 0.04, 0.07, 0.05),
    ),
    "leakage_m3_s",
    (at_most("length_m", 0.1),),
    solver="interFoam",
    simplification="axisymmetric, short transient from an air-filled gap",
)
