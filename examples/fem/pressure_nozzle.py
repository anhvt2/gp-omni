"""Radial nozzle on a cylindrical pressure vessel with a reinforcing pad.

A quarter model uses the two symmetry planes. The vessel axis is x, the nozzle axis is z,
and the cut ends carry the closed-vessel pressure thrust.

Run ``python examples/fem/pressure_nozzle.py {describe,baseline,optimize}``.
"""

import numpy as np
from common import cli, solid
from common.facets import TOLERANCE_MM, on_cylinder, on_plane, select_facets
from common.materials import STEEL
from common.mesh import build_mesh
from common.problem import at_most, design_problem, variable

INNER_RADIUS, HALF_LENGTH = 150.0, 250.0
BORE_RADIUS, NOZZLE_TOP = 40.0, 260.0
PRESSURE_MPA = 1.5


def geometry(occ, design):
    shell = INNER_RADIUS + design["shell_wall_mm"]
    nozzle = BORE_RADIUS + design["nozzle_wall_mm"]
    pad_radius = nozzle + design["pad_width_mm"]

    def along_x(radius):
        return [(3, occ.addCylinder(0, 0, 0, HALF_LENGTH, 0, 0, radius))]

    def along_z(radius, top=NOZZLE_TOP):
        return [(3, occ.addCylinder(0, 0, 0, 0, 0, top, radius))]

    pad, _ = occ.intersect(along_x(shell + design["pad_thickness_mm"]), along_z(pad_radius))
    body, _ = occ.fuse(along_x(shell), pad + along_z(nozzle))
    body, _ = occ.cut(body, along_x(INNER_RADIUS) + along_z(BORE_RADIUS, NOZZLE_TOP + 1))
    quarter = [(3, occ.addBox(0, 0, -400, 400, 400, 800))]
    occ.intersect(body, quarter)


def solve(design, mesh_scale):
    mesh = build_mesh(lambda occ: geometry(occ, design), 9.0 * mesh_scale, per_circle=24)
    vessel_wall = on_cylinder(mesh, 0, (0, 0, 0), INNER_RADIUS)
    bore = on_cylinder(
        mesh,
        2,
        (0, 0, 0),
        BORE_RADIUS,
        keep=lambda p: (p[2] > 0) & (np.hypot(p[1], p[2]) > INNER_RADIUS - TOLERANCE_MM),
    )
    end = on_plane(mesh, 0, HALF_LENGTH)
    nozzle_end = on_plane(mesh, 2, NOZZLE_TOP)
    # holds the only remaining rigid motion; the loads are in equilibrium along z
    anchor = select_facets(
        mesh, lambda p: (np.abs(p[0] - HALF_LENGTH) < TOLERANCE_MM) & (np.abs(p[2]) < 20)
    )
    fixed = [(on_plane(mesh, 0, 0), (0,)), (on_plane(mesh, 1, 0), (1,)), (anchor, (2,))]
    loads = [
        solid.pressure(np.union1d(vessel_wall, bore), PRESSURE_MPA),
        solid.total_force(mesh, end, (PRESSURE_MPA * np.pi * INNER_RADIUS**2 / 2, 0, 0)),
        solid.total_force(mesh, nozzle_end, (0, 0, PRESSURE_MPA * np.pi * BORE_RADIUS**2 / 4)),
    ]
    stress = solid.von_mises(solid.solve_static(mesh, STEEL, fixed, loads))
    return {
        "mass_kg": 4 * solid.mass_kg(mesh, STEEL),
        "stress_mpa": stress.percentile(
            99.5, keep=lambda p: (np.hypot(p[0], p[1]) < 150) & (p[2] > 100)
        ),
        "shell_stress_mpa": stress.percentile(50, keep=lambda p: (p[0] > 180) & (p[0] < 220)),
    }


PROBLEM = design_problem(
    name="pressure_nozzle",
    title="Reinforced pressure vessel nozzle",
    variables=(
        variable("shell_wall_mm", 4, 14, 9),
        variable("nozzle_wall_mm", 3, 12, 7),
        variable("pad_width_mm", 10, 60, 30),
        variable("pad_thickness_mm", 1, 12, 6),
    ),
    objective="mass_kg",
    constraints=(at_most("stress_mpa", 50),),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
