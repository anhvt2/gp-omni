"""External flows on structured meshes: plate, airfoil and bullet."""

from math import cos, pi, radians, sin

import numpy as np
from foam import measure as post
from foam.boundaries import (
    empty,
    freestream,
    wall,
)
from foam.case import cfd_case, run_plan
from foam.hex_mesh import hex_mesh
from foam.physics import flow, write_flow_case
from foam.shapes import naca_outline

from .internal import _finish

AIR_DENSITY = 1.2


def _graded(first: float, length: float, cells: int) -> np.ndarray:
    """Nodes from 0 to ``length`` growing geometrically from a first cell of ``first``."""
    if first * cells >= length:
        return np.linspace(0, length, cells + 1)
    low, high = 1.0 + 1e-9, 2.0
    for _ in range(100):
        ratio = 0.5 * (low + high)
        total = first * (ratio**cells - 1) / (ratio - 1)
        low, high = (ratio, high) if total < length else (low, ratio)
    steps = first * ratio ** np.arange(cells)
    nodes = np.concatenate([[0.0], np.cumsum(steps)])
    return nodes * length / nodes[-1]


def _airfoil_mesh(level) -> tuple[hex_mesh, float, float]:
    """O-grid around a NACA 0012 section of unit chord in the x-y plane."""
    chord, radius, span = 1.0, 15.0, 0.1
    outline = naca_outline(chord, points=level.cells(60))
    half = level.cells(60)
    upper, lower = outline[:half], outline[half:]
    # start at the trailing edge, run along the upper side to the nose and back below
    loop = np.vstack([upper[::-1], lower[::-1], upper[-1:]])
    count = len(loop) - 1
    angle = 2 * pi * np.arange(count + 1) / count
    far = np.column_stack([0.5 + radius * np.cos(angle), radius * np.sin(angle)])
    t = _graded(2e-3, radius, level.cells(60)) / radius
    plane = loop[:, None, :] * (1 - t[None, :, None]) + far[:, None, :] * t[None, :, None]
    points = np.empty((count + 1, len(t), 2, 3))
    points[..., 0] = plane[:, :, None, 0]
    points[..., 1] = plane[:, :, None, 1]
    points[..., 2] = np.array([0.0, span])[None, None, :]
    mesh = hex_mesh()
    mesh.add_block(
        points,
        {"i0": "cut", "i1": "cut", "j0": "airfoil", "j1": "farfield", "k0": "front",
         "k1": "back"},
    )  # fmt: skip
    return mesh, chord, span


def _airfoil(case, level, angle_deg: float, speed: float, time_step: float = 0.0):
    mesh, chord, span = _airfoil_mesh(level)
    angle = radians(angle_deg)
    velocity = (speed * cos(angle), speed * sin(angle), 0.0)
    patches = _finish(
        case,
        mesh,
        {"airfoil": wall(), "farfield": freestream(velocity), "front": empty(), "back": empty()},
    )
    functions = post.forces("airfoil", ["airfoil"], AIR_DENSITY, (0.25, 0, 0))
    if time_step:
        functions |= post.probes("pressure_probes", ((1.02, 0.03, 0.05), (0.9, 0.12, 0.05)))
    solver = "pimpleFoam" if time_step else "simpleFoam"
    write_flow_case(
        case,
        patches,
        flow(
            solver, velocity=speed, length=0.1 * chord, iterations=level.iterations,
            time_step=time_step, time_steps=level.time_steps,
            second_order=level.resolution > 1, functions=functions,
        ),
    )  # fmt: skip

    def measure(directory):
        force = post.last_force(directory, "airfoil")
        dynamic = 0.5 * AIR_DENSITY * speed**2 * chord * span
        drag = float(force[0] * cos(angle) + force[1] * sin(angle)) / dynamic
        lift = float(force[1] * cos(angle) - force[0] * sin(angle)) / dynamic
        metrics = {
            "lift_coefficient": lift,
            "drag_coefficient": drag,
            "lift_to_drag": lift / drag if abs(drag) > 1e-9 else 0.0,
        }
        if time_step:
            metrics["pressure_fluctuation_pa"] = AIR_DENSITY * post.probe_fluctuation(
                directory, "pressure_probes"
            )
        return metrics

    return run_plan(solve=[[solver]], measure=measure)


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    return cfd_case(
        name, "external", title, "structured", solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip
