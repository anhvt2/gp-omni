"""Welded steel lifting lug under an inclined pin load.

The pin is a cosine bearing pressure on the loaded half of the hole; contact is not
modeled. The welded base is fixed.

Run ``python examples/fem/lifting_lug.py {describe,baseline,optimize}``.
"""

import numpy as np
from common import cli, solid
from common.facets import on_cylinder, on_plane
from common.materials import STEEL
from common.mesh import build_mesh, prism
from common.problem import at_most, design_problem, variable

HOLE_RADIUS = 12.5
LOAD_N = 40_000.0
SLING_DEG = 30.0  # from vertical, in the plane of the plate
DIRECTION = np.array([np.sin(np.radians(SLING_DEG)), 0.0, np.cos(np.radians(SLING_DEG))])


def geometry(occ, design):
    thickness, radius = design["thickness_mm"], design["outer_radius_mm"]
    height, half = design["hole_height_mm"], design["base_width_mm"] / 2
    # the flanks leave the base corners and touch the rounded top tangentially
    angle = np.arctan2(-height, half) + np.arccos(radius / np.hypot(half, height))
    x, z = radius * np.cos(angle), height + radius * np.sin(angle)
    corners = [(-half, 0, 0), (half, 0, 0), (x, 0, z), (-x, 0, z)]
    flanks = prism(occ, corners, (0, thickness, 0))
    top = [(3, occ.addCylinder(0, 0, height, 0, thickness, 0, radius))]
    body, _ = occ.fuse(flanks, top)
    hole = [(3, occ.addCylinder(0, -1, height, 0, thickness + 2, 0, HOLE_RADIUS))]
    occ.cut(body, hole)


def prepare_mesh(design, mesh_scale):
    return build_mesh(lambda occ: geometry(occ, design), 4.5 * mesh_scale, per_circle=24)


def solve(design, mesh_scale):
    return solve_mesh(design, prepare_mesh(design, mesh_scale))


def solve_mesh(design, mesh, fields=None):
    height = design["hole_height_mm"]
    hole_axis = (0, 0, height)
    base = on_plane(mesh, 2, 0)
    hole = on_cylinder(mesh, 1, hole_axis, HOLE_RADIUS)
    load = solid.bearing_force(mesh, hole, LOAD_N * DIRECTION, hole_axis, HOLE_RADIUS)
    result = solid.solve_static(mesh, STEEL, [solid.clamp(base)], [load])
    if fields is not None:
        fields.update(bearing=result)
    return {
        "mass_kg": solid.mass_kg(mesh, STEEL),
        "stress_mpa": solid.von_mises(result).percentile(99, keep=lambda p: p[2] > 3),
        "deflection_mm": float(np.linalg.norm(solid.mean_displacement(result, hole))),
    }


PROBLEM = design_problem(
    name="lifting_lug",
    title="Lifting lug",
    variables=(
        variable("thickness_mm", 8, 25, 16),
        variable("outer_radius_mm", 22, 45, 32),
        variable("hole_height_mm", 50, 90, 65),
        variable("base_width_mm", 50, 110, 80),
    ),
    objective="mass_kg",
    constraints=(at_most("stress_mpa", 170), at_most("deflection_mm", 0.05)),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
