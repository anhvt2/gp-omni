"""Shared wind-tunnel set-up of the bodies meshed with ``snappyHexMesh``."""

from collections.abc import Callable, Mapping
from pathlib import Path

from foam import measure as post
from foam.boundaries import boundary, pressure_outlet, slip, symmetry, velocity_inlet, wall
from foam.case import mesh_level, run_plan
from foam.physics import flow, write_flow_case
from foam.snappy import domain, surface, write_snappy

DENSITY = 1.2
AMBIENT_PRESSURE = 101325.0
AMBIENT_TEMPERATURE = 293.15


def wind_tunnel(
    case: Path,
    level: mesh_level,
    surfaces: Mapping[str, surface],
    lower: tuple,
    upper: tuple,
    cell: float,
    speed: float,
    length: float,
    area: float,
    half: bool = False,
    ground: bool = True,
    refine: list | None = None,
    centre: tuple = (0.0, 0.0, 0.0),
    extra: Mapping[str, float] | None = None,
    inside: tuple | None = None,
    overrides: Mapping[str, boundary] | None = None,
    functions: Mapping[str, object] | None = None,
    derive: Callable[[Path, dict], dict] | None = None,
    time_step: float = 0.0,
    compressible: bool = False,
    libraries: tuple[str, ...] = (),
) -> run_plan:
    """Flow along +x past ``surfaces``; coefficients use the full frontal ``area``.

    ``half`` models ``y >= 0`` only with a symmetry plane, and doubles the loads.
    ``time_step`` selects a transient run, ``overrides`` replaces patch conditions and
    ``derive`` adds metrics computed from the output.
    """
    sides = {
        "x0": "inlet", "x1": "outlet", "y0": "left", "y1": "right",
        "z0": "ground", "z1": "top",
    }  # fmt: skip
    patches = {
        "inlet": velocity_inlet(
            (speed, 0, 0), temperature=AMBIENT_TEMPERATURE if compressible else None
        ),
        "outlet": pressure_outlet(AMBIENT_PRESSURE if compressible else 0.0),
        "left": symmetry() if half else slip(),
        "right": slip(),
        "ground": wall() if ground else slip(),
        "top": slip(),
    }
    patches |= {name: wall() for name in surfaces}
    patches |= dict(overrides or {})
    if inside is None:
        inside = tuple(
            low + fraction * (high - low)
            for low, high, fraction in zip(lower, upper, (0.0871, 0.7313, 0.8137))
        )
    box = domain(lower, upper, cell, sides, refine or [])
    mesh = write_snappy(case, box, surfaces, patches, inside, level)
    objects = post.flux("inlet", "outlet") | post.forces(
        "body",
        tuple(surfaces),
        None if compressible else DENSITY,
        centre,
        AMBIENT_PRESSURE if compressible else 0.0,
    )
    solver = "simpleFoam"
    if time_step:
        solver = "rhoPimpleFoam" if compressible else "pimpleFoam"
    model = flow(
        solver,
        velocity=speed,
        length=length,
        density=DENSITY,
        iterations=level.iterations,
        time_step=time_step,
        time_steps=level.time_steps,
        temperature=AMBIENT_TEMPERATURE,
        second_order=level.resolution > 1,
        functions=objects | dict(functions or {}),
        libraries=libraries,
        initial_velocity=(speed, 0.0, 0.0),
    )
    write_flow_case(case, patches, model)

    def measure(directory: Path) -> dict[str, float]:
        scale = (2.0 if half else 1.0) / (0.5 * DENSITY * speed**2 * area)
        force = post.last_force(directory, "body")
        moment = post.last_moment(directory, "body")
        drag = float(force[0] * scale)
        lift = float(force[2] * scale)
        metrics = {
            "flow_mismatch" if time_step else "flow_imbalance": post.imbalance(
                directory, "inlet", "outlet"
            ),
            "drag_coefficient": drag,
            "lift_coefficient": lift,
            "downforce_coefficient": -lift,
            "lift_to_drag": lift / drag if abs(drag) > 1e-9 else 0.0,
            "downforce_to_drag": -lift / drag if abs(drag) > 1e-9 else 0.0,
            "base_moment_coefficient": float(
                abs(moment[1]) * scale / max(length, 1e-9)
            ),
            **(extra or {}),
        }
        return metrics | (derive(directory, metrics) if derive else {})

    return run_plan(mesh=mesh, solve=[[solver]], measure=measure)
