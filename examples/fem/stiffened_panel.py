"""Edge-clamped aluminum panel with three longitudinal ribs and one cross rib.

Run ``python examples/fem/stiffened_panel.py {describe,baseline,optimize}``.
"""

import numpy as np
from common import cli, solid
from common.facets import on_plane
from common.materials import ALUMINUM
from common.mesh import build_mesh
from common.problem import at_least, at_most, design_problem, variable

LENGTH, WIDTH = 300.0, 200.0
PRESSURE_MPA = 0.002


def geometry(occ, design):
    skin, height = design["skin_mm"], design["rib_height_mm"]
    thickness, spacing = design["rib_thickness_mm"], design["rib_spacing_mm"]
    plate = [(3, occ.addBox(0, 0, 0, LENGTH, WIDTH, skin))]
    z, depth = skin / 2, height + skin / 2
    ribs = [
        (3, occ.addBox(0, WIDTH / 2 + offset - thickness / 2, z, LENGTH, thickness, depth))
        for offset in (-spacing, 0, spacing)
    ]
    ribs.append((3, occ.addBox((LENGTH - thickness) / 2, 0, z, thickness, WIDTH, depth)))
    occ.fuse(plate, ribs)


def solve(design, mesh_scale):
    mesh = build_mesh(lambda occ: geometry(occ, design), 7.0 * mesh_scale)
    sides = ((0, 0), (0, LENGTH), (1, 0), (1, WIDTH))
    edges = np.unique(np.concatenate([on_plane(mesh, axis, value) for axis, value in sides]))
    load = solid.pressure(on_plane(mesh, 2, 0), PRESSURE_MPA)
    panel = solid.structure(mesh, ALUMINUM, [solid.clamp(edges)])
    result = panel.deform([load])
    vertical = result.displacement[result.basis.split_indices()[2]]
    return {
        "mass_kg": solid.mass_kg(mesh, ALUMINUM),
        "frequency_hz": float(panel.frequencies()[0]),
        "deflection_mm": float(np.abs(vertical).max()),
    }


PROBLEM = design_problem(
    name="stiffened_panel",
    title="Rib-stiffened panel",
    variables=(
        variable("skin_mm", 0.8, 3, 1.5),
        variable("rib_height_mm", 4, 25, 12),
        variable("rib_thickness_mm", 1, 4, 2),
        variable("rib_spacing_mm", 30, 80, 55),
    ),
    objective="mass_kg",
    constraints=(at_least("frequency_hz", 900), at_most("deflection_mm", 0.015)),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
