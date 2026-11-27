"""Turbulent flat plate against a drag correlation."""

import numpy as np
from foam import measure as post
from foam.boundaries import empty, pressure_outlet, slip, symmetry, velocity_inlet, wall
from foam.case import at_most, run_plan, variable
from foam.hex_mesh import hex_mesh, lattice
from foam.physics import flow, write_flow_case

from ..external_structured import AIR_DENSITY, _case, _graded
from ..internal import _finish

VISCOSITY = 1.5e-5


def flat_plate(case, design, level):
    speed, first = design["speed_m_s"], design["first_cell_m"]
    length, height, span = 2.0, 1.0, 0.1
    lead = -_graded(0.01, 0.3, level.cells(16))[::-1]
    plate = _graded(0.01, length, level.cells(60))
    x = np.concatenate([lead[:-1], plate])
    y = _graded(first, height, level.cells(50))
    grid = lattice(x, y, np.array([0.0, span]), lambda a, b, c: (a, b, c))
    middle = 0.5 * (x[1:] + x[:-1])
    bottom = np.where(middle > 0, "plate", "upstream").astype(object)
    mesh = hex_mesh()
    mesh.add_block(
        grid,
        {"i0": "inlet", "i1": "outlet", "j0": bottom[:, None], "j1": "top",
         "k0": "front", "k1": "back"},
    )  # fmt: skip
    patches = _finish(
        case,
        mesh,
        {
            "inlet": velocity_inlet((speed, 0, 0)), "outlet": pressure_outlet(),
            "plate": wall(), "upstream": symmetry(), "top": slip(), "front": empty(),
            "back": empty(),
        },
    )  # fmt: skip
    functions = post.flux("inlet", "outlet") | post.forces("plate", ["plate"], AIR_DENSITY)
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam", velocity=speed, length=0.1, iterations=level.iterations,
            second_order=level.resolution > 1, functions=functions,
            initial_velocity=(speed, 0.0, 0.0),
        ),
    )  # fmt: skip
    cells = (len(x) - 1) * (len(y) - 1)
    reynolds = speed * length / VISCOSITY
    correlation = 0.074 / reynolds**0.2

    def measure(directory):
        drag = float(post.last_force(directory, "plate")[0])
        coefficient = drag / (0.5 * AIR_DENSITY * speed**2 * length * span)
        return {
            # the top boundary also passes flow, so inlet and outlet are not compared
            "drag_coefficient": coefficient,
            "drag_error": abs(coefficient - correlation) / correlation,
            "cell_count": float(cells),
        }

    return run_plan(solve=[["simpleFoam"]], measure=measure)


CASE = _case(
    "flat_plate",
    "Turbulent flat plate against a drag correlation",
    flat_plate,
    (variable("speed_m_s", 30.0, 70.0, 50.0), variable("first_cell_m", 2e-4, 3e-3, 1e-3)),
    "drag_error",
    (at_most("cell_count"),),
    positive=("drag_coefficient",),
    simplification="operating variables only, a pipeline check",
)
