"""Steel connecting rod with an I-section shank.

The big-end bore is fixed. A tensile bearing load and a separate lateral load act on the
small-end bore. Contact, fatigue and buckling are not modeled.

Run ``python examples/fem/connecting_rod.py {describe,baseline,optimize}``.
"""

import numpy as np
from common import cli, solid
from common.facets import on_cylinder
from common.materials import STEEL
from common.mesh import build_mesh
from common.problem import at_most, design_problem, variable

CENTERS = 130.0
BIG_BORE, SMALL_BORE = 16.0, 9.0
TENSION_N, LATERAL_N = 15_000.0, 500.0


def geometry(occ, design):
    width, flange = design["shank_width_mm"], design["flange_mm"]
    web, depth = design["web_mm"], design["depth_mm"]
    big, small = design["big_radius_mm"], design["small_radius_mm"]
    z = -depth / 2
    ends = [
        (3, occ.addCylinder(0, 0, z, 0, 0, depth, big)),
        (3, occ.addCylinder(CENTERS, 0, z, 0, 0, depth, small)),
    ]
    shank = [(3, occ.addBox(0, -width / 2, z, CENTERS, width, depth))]
    body, _ = occ.fuse(shank, ends)
    start, stop = big + 3, CENTERS - small - 3
    inner = width - 2 * flange
    voids = [
        (3, occ.addBox(start, -inner / 2, side, stop - start, inner, depth))
        for side in (web / 2, -web / 2 - depth)
    ]
    voids += [
        (3, occ.addCylinder(x, 0, z - 1, 0, 0, depth + 2, radius))
        for x, radius in ((0, BIG_BORE), (CENTERS, SMALL_BORE))
    ]
    occ.cut(body, voids)


def prepare_mesh(design, mesh_scale):
    return build_mesh(lambda occ: geometry(occ, design), 3.5 * mesh_scale, per_circle=20)


def solve(design, mesh_scale):
    return solve_mesh(design, prepare_mesh(design, mesh_scale))


def solve_mesh(design, mesh, fields=None):
    small_axis = (CENTERS, 0, 0)
    big = on_cylinder(mesh, 2, (0, 0, 0), BIG_BORE)
    small = on_cylinder(mesh, 2, small_axis, SMALL_BORE)
    rod = solid.structure(mesh, STEEL, [solid.clamp(big)])
    bearing = solid.bearing_force(mesh, small, (TENSION_N, 0, 0), small_axis, SMALL_BORE)
    tension = rod.deform([bearing])
    side = solid.total_force(mesh, small, (0, 0, LATERAL_N))
    bending = rod.deform([side])
    if fields is not None:
        fields.update(tension=tension, lateral=bending)
    return {
        "mass_kg": solid.mass_kg(mesh, STEEL),
        "stress_mpa": solid.von_mises(tension).percentile(
            99, keep=lambda p: np.hypot(p[0], p[1]) > BIG_BORE + 1.5
        ),
        "elongation_mm": float(solid.mean_displacement(tension, small)[0]),
        "lateral_deflection_mm": float(solid.mean_displacement(bending, small)[2]),
    }


PROBLEM = design_problem(
    name="connecting_rod",
    title="I-section connecting rod",
    variables=(
        variable("shank_width_mm", 12, 24, 18),
        variable("flange_mm", 1.5, 4, 3),
        variable("web_mm", 2, 6, 4),
        variable("depth_mm", 10, 22, 16),
        variable("big_radius_mm", 20, 28, 24),
        variable("small_radius_mm", 13, 18, 15),
    ),
    objective="mass_kg",
    constraints=(
        at_most("stress_mpa", 230),
        at_most("elongation_mm", 0.07),
        at_most("lateral_deflection_mm", 0.6),
    ),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
