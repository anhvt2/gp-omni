"""Wall-mounted L bracket with a central gusset and four lightening holes.

Run ``python examples/fem/l_bracket.py {describe,baseline,optimize}``.
"""

from common import cli, solid
from common.facets import on_plane
from common.materials import ALUMINUM
from common.mesh import build_mesh, prism
from common.problem import at_most, design_problem, variable

REACH, DROP, WIDTH = 80.0, 80.0, 50.0
LOAD_N = 600.0
HOLES = ((40.0, 12.0), (62.0, 12.0), (40.0, 38.0), (62.0, 38.0))


def geometry(occ, design):
    flat, upright = design["horizontal_thickness_mm"], design["vertical_thickness_mm"]
    web, leg = design["gusset_thickness_mm"], design["gusset_size_mm"]
    parts = [
        (3, occ.addBox(0, 0, -flat, REACH, WIDTH, flat)),
        (3, occ.addBox(0, 0, -DROP, upright, WIDTH, DROP)),
    ]
    corners = [(upright / 2, -flat / 2), (upright + leg, -flat / 2), (upright / 2, -flat - leg)]
    gusset = [(x, (WIDTH - web) / 2, z) for x, z in corners]
    parts += prism(occ, gusset, (0, web, 0))
    body, _ = occ.fuse(parts[:1], parts[1:])
    radius = design["hole_radius_mm"]
    holes = [(3, occ.addCylinder(x, y, -flat - 1, 0, 0, flat + 2, radius)) for x, y in HOLES]
    occ.cut(body, holes)


def solve(design, mesh_scale):
    mesh = build_mesh(lambda occ: geometry(occ, design), 3.5 * mesh_scale)
    wall = on_plane(mesh, 0, 0)
    tip = on_plane(mesh, 0, REACH)
    load = solid.total_force(mesh, tip, (0, 0, -LOAD_N))
    result = solid.solve_static(mesh, ALUMINUM, [solid.clamp(wall)], [load])
    return {
        "mass_kg": solid.mass_kg(mesh, ALUMINUM),
        "deflection_mm": float(-solid.mean_displacement(result, tip)[2]),
        "stress_mpa": solid.von_mises(result).percentile(99, keep=lambda p: p[0] > 1),
    }


PROBLEM = design_problem(
    name="l_bracket",
    title="Gusseted L bracket",
    variables=(
        variable("horizontal_thickness_mm", 3, 10, 7),
        variable("vertical_thickness_mm", 3, 10, 7),
        variable("gusset_thickness_mm", 2, 10, 5),
        variable("gusset_size_mm", 15, 55, 35),
        variable("hole_radius_mm", 3, 7, 4),
    ),
    objective="mass_kg",
    constraints=(at_most("deflection_mm", 0.6), at_most("stress_mpa", 70)),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
