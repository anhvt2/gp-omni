"""Branched and shaped internal flows: Gmsh surfaces meshed with ``snappyHexMesh``."""

from pathlib import Path

import numpy as np
from foam import measure as post
from foam.boundaries import wall
from foam.case import cfd_case
from foam.physics import flow, write_flow_case
from foam.snappy import domain, surface, write_snappy

WATER = {"density": 1000.0, "viscosity": 1e-6}
AIR_DENSITY = 1.2


def _snappy_fluid(case, level, surfaces: dict, patches: dict, cell: float, inside, depth=(1, 2)):
    """Mesh the region enclosed by ``surfaces`` inside a slightly larger box."""
    if set(surfaces) != set(patches):
        raise ValueError(f"surfaces {sorted(surfaces)} do not match {sorted(patches)}")
    everything = np.concatenate([t.reshape(-1, 3) for t in surfaces.values()])
    low, high = everything.min(axis=0), everything.max(axis=0)
    margin = 1.37 * cell
    box = domain(
        tuple(low - margin),
        tuple(high + 1.11 * margin),
        cell,
        {side: "outside" for side in ("x0", "x1", "y0", "y1", "z0", "z1")},
    )
    all_patches = patches | {"outside": wall()}
    commands = write_snappy(
        case,
        box,
        {n: surface(t, depth) for n, t in surfaces.items()},
        all_patches,
        inside,
        level,
    )
    return commands, patches


def _steady(case, patches, speed, length, level, functions, **options):
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam",
            velocity=speed,
            length=length,
            iterations=level.iterations,
            second_order=level.resolution > 1,
            functions=functions,
            **options,
        ),
    )


def _total(directory: Path, patch: str) -> float:
    return post.last_surface(directory, f"total_p_{patch}")


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    return cfd_case(
        name, "internal", title, "snappy", solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip
