"""Ventilated room with a seated occupant."""

import numpy as np
from foam import measure as post
from foam.boundaries import pressure_outlet, velocity_inlet, wall
from foam.case import at_most, run_plan, variable
from foam.hex_mesh import box_grid, hex_mesh
from foam.physics import flow, write_flow_case

from ..internal import _case, _finish


def human_comfort(case, design, level):
    """A 4 x 3 x 2.7 m room with one seated heat source, supply low and exhaust high."""
    lower, upper = (0.0, 0.0, 0.0), (4.0, 3.0, 2.7)
    cells = [level.cells(n) for n in (40, 30, 27)]
    grid = box_grid(lower, upper, cells)
    centre = 0.25 * (grid[1:, 1:, 1:] + grid[:-1, :-1, 1:] + grid[1:, :-1, :-1]
                     + grid[:-1, 1:, :-1])  # fmt: skip
    x, y, z = np.moveaxis(centre, -1, 0)
    person = (np.abs(x - 2.4) < 0.2) & (np.abs(y - 1.5) < 0.25) & (z < 1.2)
    occupied = (x > 1.0) & (x < 3.4) & (y > 0.5) & (y < 2.5) & (z < 1.8)
    zone = np.where(occupied & ~person, "occupied", None).astype(object)
    # cell-face centres of the two end walls
    face_y = 0.5 * (grid[0, 1:, 0, 1] + grid[0, :-1, 0, 1])
    face_z = 0.5 * (grid[0, 0, 1:, 2] + grid[0, 0, :-1, 2])
    across = np.abs(face_y - 1.5)[:, None] < 0.5
    vent = design["vent_height_m"]
    supply = np.where(across & (np.abs(face_z - vent) < 0.15)[None, :], "inlet", "walls")
    exhaust = np.where(across & (np.abs(face_z - 2.5) < 0.12)[None, :], "outlet", "walls")
    mesh = hex_mesh()
    mesh.add_block(
        grid,
        {"i0": supply.astype(object), "i1": exhaust.astype(object), "k0": "floor", "k1": "walls",
         "j0": "walls", "j1": "walls"},
        keep=~person,
        exposed="person",
        zone=zone,
    )  # fmt: skip
    speed = design["inlet_speed_m_s"]
    supply_temperature = design["supply_temperature_k"]
    patches = _finish(
        case,
        mesh,
        {
            "inlet": velocity_inlet((speed, 0, 0), temperature=supply_temperature),
            "outlet": pressure_outlet(101325.0), "walls": wall(), "floor": wall(296.0),
            "person": wall(307.0),
        },
    )  # fmt: skip
    functions = post.flux("inlet", "outlet")
    functions |= {
        "speed": {"type": "mag", "libs": "(fieldFunctionObjects)", "field": "U",
                  "result": "magU", "writeControl": "none", "log": False}
    }  # fmt: skip
    functions |= post.volume_value("occupied_temperature", "volAverage", "T", "occupied")
    functions |= post.volume_value("occupied_speed", "volAverage", "magU", "occupied")
    write_flow_case(
        case,
        patches,
        flow(
            "buoyantSimpleFoam",
            velocity=speed,
            length=0.3,
            iterations=level.iterations,
            temperature=295.0,
            functions=functions,
        ),
    )

    def measure(directory):
        temperature = post.last_volume(directory, "occupied_temperature")
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "occupied_temperature_k": temperature,
            "comfort_deviation_k": abs(temperature - 295.65),
            "draught_speed_m_s": post.last_volume(directory, "occupied_speed"),
        }

    return run_plan(solve=[["buoyantSimpleFoam"]], measure=measure)


CASE = _case(
    "human_comfort",
    "Ventilated room with a seated occupant",
    human_comfort,
    (
        variable("inlet_speed_m_s", 0.15, 0.6, 0.3),
        variable("supply_temperature_k", 288.0, 296.0, 292.0),
        variable("vent_height_m", 0.3, 1.8, 0.5),
    ),
    "comfort_deviation_k",
    (at_most("draught_speed_m_s", 0.3),),
    solver="buoyantSimpleFoam",
    simplification="box manikin, no radiation or humidity",
)
