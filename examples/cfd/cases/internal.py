"""Internal and process flows on structured meshes."""

from math import pi, radians

import numpy as np
from foam import measure as post
from foam.boundaries import (
    flow_inlet,
    pressure_outlet,
    wall,
)
from foam.case import cfd_case, run_plan
from foam.ducts import sweep, u_bend
from foam.hex_mesh import hex_mesh, lattice
from foam.physics import flow, write_flow_case

WATER = {"density": 1000.0, "viscosity": 1e-6}
HALF_WEDGE = radians(2.5)
WEDGES = 72  # wedges in a full turn


def _finish(case, mesh: hex_mesh, patches: dict) -> dict:
    """Check and write a structured mesh; return the conditions of the patches it has."""
    built = mesh.build()
    built.require_positive_volumes()
    present = {n: patches[n] for n in built.patch_names()}
    built.write(case, {n: b.mesh_entry() for n, b in present.items()})
    return present


def _wedge_grid(x, inner, outer, radial: int) -> np.ndarray:
    """Axisymmetric lattice about x between the radii ``inner(x)`` and ``outer(x)``."""
    t = np.linspace(0, 1, radial + 1)
    t = 0.5 * t + 0.5 * (0.5 - 0.5 * np.cos(pi * t))
    r = np.asarray(inner)[:, None] * (1 - t) + np.asarray(outer)[:, None] * t
    angle = np.array([-HALF_WEDGE, HALF_WEDGE])
    points = np.empty((len(x), len(t), 2, 3))
    points[..., 0] = np.asarray(x)[:, None, None]
    points[..., 1] = r[:, :, None] * np.cos(angle)
    points[..., 2] = r[:, :, None] * np.sin(angle)
    return points


def _bend(case, level, bend_radius: float, radius_at, flow_rate: float, diameter: float):
    """Water through a 180 degree bend; ``radius_at`` takes the path fraction."""
    path, total = u_bend(0.4, bend_radius, 0.8)
    mesh = hex_mesh()
    mesh.add_block(
        sweep(path, radius_at, level.cells(80), level.cells(14, 8)),
        {"i0": "inlet", "i1": "outlet"},
    )
    patches = _finish(
        case,
        mesh,
        {"inlet": flow_inlet(flow_rate), "outlet": pressure_outlet(), "walls": wall()},
    )
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam",
            velocity=flow_rate / (pi * diameter**2 / 4),
            length=diameter,
            iterations=level.iterations,
            second_order=level.resolution > 1,
            functions=post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet"),
            **WATER,
        ),
    )
    s = np.linspace(0, 1, 801)
    volume = float(np.trapezoid(pi * radius_at(s) ** 2, s * total))

    def measure(directory):
        loss = post.last_surface(directory, "total_p_inlet") - post.last_surface(
            directory, "total_p_outlet"
        )
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "pressure_loss_pa": WATER["density"] * loss,
            "volume_m3": volume,
        }

    return run_plan(solve=[["simpleFoam"]], measure=measure)


def _polar_grid(r, theta, z) -> np.ndarray:
    return lattice(r, theta, z, lambda a, b, c: (a * np.cos(b), a * np.sin(b), c))


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    return cfd_case(
        name, "internal", title, "structured", solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip
