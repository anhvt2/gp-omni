"""Sliding-mesh siren."""

from math import ceil, pi, radians

import numpy as np
from foam import measure as post
from foam.boundaries import flow_inlet, interface, moving_wall, pressure_outlet, wall
from foam.case import at_most, run_plan, variable
from foam.hex_mesh import hex_mesh
from foam.physics import flow, write_flow_case

from ..internal import _case, _finish, _polar_grid


def mechanical_siren(case, design, level):
    """A slotted rotor chopping the flow just ahead of a slotted stator."""
    hub, tip = 0.03, 0.08
    ports = round(design["port_count"])
    gap, omega = design["gap_m"], design["speed_rad_s"]
    rotor_start, thickness, lead = 0.1, 0.008, 0.003
    rotor_face = rotor_start - lead  # sliding faces stay clear of the solid discs
    rotor_end = rotor_start + thickness + gap / 2
    stator_start = rotor_start + thickness + gap
    per_port = max(8, level.cells(96) // ports)
    theta = np.linspace(0, 2 * pi, ports * per_port + 1)
    r = np.linspace(hub, tip, level.cells(10) + 1)
    open_slot = (np.arange(len(theta) - 1) % per_port) < per_port // 2

    def block(z_nodes, solid_range):
        zc = 0.5 * (z_nodes[1:] + z_nodes[:-1])
        solid = (zc > solid_range[0]) & (zc < solid_range[1])
        covered = (r[:-1] >= hub + 0.25 * (tip - hub)) & (r[1:] <= tip - 0.1 * (tip - hub))
        mask = covered[:, None, None] & ~open_slot[None, :, None] & solid[None, None, :]
        return _polar_grid(r, theta, z_nodes), ~mask

    def axial(*breaks):
        parts = [np.linspace(a, b, level.cells(n) + 1) for a, b, n in breaks]
        return np.unique(np.concatenate(parts))

    pieces = (
        (axial((0.0, rotor_face, 16)), (-1.0, -1.0), 0, {"k0": "inlet", "k1": "front_static"}),
        (
            axial((rotor_face, rotor_start, 2),
                  (rotor_start, rotor_start + thickness, 4),
                  (rotor_start + thickness, rotor_end, 2)),
            (rotor_start, rotor_start + thickness), 1,
            {"k0": "front_rotor", "k1": "rear_rotor"},
        ),
        (
            axial((rotor_end, stator_start, 2),
                  (stator_start, stator_start + thickness, 4),
                  (stator_start + thickness, 0.25, 18)),
            (stator_start, stator_start + thickness), 0,
            {"k0": "rear_static", "k1": "outlet"},
        ),
    )  # fmt: skip
    mesh = hex_mesh()
    for z_nodes, solid_range, region, caps in pieces:
        points, keep = block(z_nodes, solid_range)
        walls = "rotor_walls" if region else "walls"
        mesh.add_block(
            points, {"i0": walls, "i1": walls, **caps}, keep=keep, exposed=walls, region=region,
            zone="rotor" if region else None,
        )  # fmt: skip
    rate = 0.08
    patches = _finish(
        case,
        mesh,
        {
            "inlet": flow_inlet(rate), "outlet": pressure_outlet(), "walls": wall(),
            "rotor_walls": moving_wall(),
            "front_static": interface("front_rotor"), "front_rotor": interface("front_static"),
            "rear_static": interface("rear_rotor"), "rear_rotor": interface("rear_static"),
        },
    )  # fmt: skip
    radius = 0.7 * tip
    functions = post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet")
    functions |= post.probes("pressure_probes", ((radius, 0.003, 0.18), (0.003, radius, 0.05)))
    step = ceil(1e6 * radians(1.5) / omega) * 1e-6
    write_flow_case(
        case,
        patches,
        flow(
            "pimpleFoam",
            velocity=rate / (pi * (tip**2 - hub**2)),
            length=0.02,
            time_step=step,
            time_steps=level.time_steps,
            motion={
                "dynamicFvMesh": "dynamicMotionSolverFvMesh",
                "motionSolverLibs": ("fvMotionSolvers",),
                "motionSolver": "solidBody",
                "cellZone": "rotor",
                "solidBodyMotionFunction": "rotatingMotion",
                "rotatingMotionCoeffs": {"origin": (0, 0, 0), "axis": (0, 0, 1), "omega": omega},
            },
            functions=functions,
        ),
    )

    def measure(directory):
        drop = post.last_surface(directory, "total_p_inlet") - post.last_surface(
            directory, "total_p_outlet"
        )
        return {
            "flow_mismatch": post.imbalance(directory, "inlet", "outlet"),
            "pressure_amplitude_pa": 1.2 * post.probe_fluctuation(directory, "pressure_probes"),
            "pressure_drop_pa": 1.2 * drop,
        }

    return run_plan(solve=[["pimpleFoam"]], measure=measure)


CASE = _case(
    "mechanical_siren",
    "Sliding-mesh siren",
    mechanical_siren,
    (
        variable("port_count", 4.0, 10.0, 6.0),
        variable("gap_m", 0.002, 0.008, 0.004),
        variable("speed_rad_s", 200.0, 500.0, 314.0),
    ),
    "pressure_amplitude_pa",
    (at_most("pressure_drop_pa", 0.5),),
    maximize=True,
    solver="pimpleFoam",
    simplification="probe pressures only, short transient",
)
