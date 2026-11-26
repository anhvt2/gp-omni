"""Cantilevered sandwich beam with a pyramidal strut core between two face sheets.

Run ``python examples/fem/lattice_beam.py {describe,baseline,optimize}``.
"""

import numpy as np
from common import cli, solid
from common.facets import on_plane
from common.materials import ALUMINUM
from common.mesh import build_mesh
from common.problem import at_least, at_most, design_problem, variable

LENGTH, WIDTH = 100.0, 20.0
CELLS = 5
INSET = 2.5  # of the strut feet from the cell edges
LOAD_N = 100.0


def geometry(occ, design):
    diameter, core = design["strut_diameter_mm"], design["core_height_mm"]
    bottom, top = design["bottom_face_mm"], design["top_face_mm"]
    total = bottom + core + top
    pitch = LENGTH / CELLS
    struts = []
    for cell in range(CELLS):
        apex = np.array([(cell + 0.5) * pitch, WIDTH / 2, bottom + core])
        for x in (cell * pitch + INSET, (cell + 1) * pitch - INSET):
            for y in (INSET, WIDTH - INSET):
                foot = np.array([x, y, bottom])
                axis = apex - foot
                extra = diameter * axis / np.linalg.norm(axis)  # reach into both faces
                struts.append(
                    (3, occ.addCylinder(*(foot - extra), *(axis + 2 * extra), diameter / 2))
                )
    # trim the strut ends at the mid-planes of the faces
    slab = [(3, occ.addBox(0, 0, bottom / 2, LENGTH, WIDTH, total - (bottom + top) / 2))]
    trimmed, _ = occ.intersect(struts, slab)
    faces = [
        (3, occ.addBox(0, 0, 0, LENGTH, WIDTH, bottom)),
        (3, occ.addBox(0, 0, bottom + core, LENGTH, WIDTH, top)),
    ]
    occ.fuse(faces, trimmed)


def solve(design, mesh_scale):
    mesh = build_mesh(
        lambda occ: geometry(occ, design),
        2.5 * mesh_scale,
        per_circle=8,
        boolean_tolerance=1e-5,  # four equal struts meet at each apex
    )
    tip = on_plane(mesh, 0, LENGTH)
    load = solid.total_force(mesh, tip, (0, 0, -LOAD_N))
    beam = solid.structure(mesh, ALUMINUM, [solid.clamp(on_plane(mesh, 0, 0))])
    result = beam.deform([load])
    return {
        "mass_kg": solid.mass_kg(mesh, ALUMINUM),
        "deflection_mm": float(-solid.mean_displacement(result, tip)[2]),
        "stress_mpa": solid.von_mises(result).percentile(99, keep=lambda p: p[0] > 3),
        "frequency_hz": float(beam.frequencies()[0]),
    }


PROBLEM = design_problem(
    name="lattice_beam",
    title="Lattice-core sandwich beam",
    variables=(
        variable("strut_diameter_mm", 1.2, 3.5, 2.2),
        variable("core_height_mm", 6, 16, 10),
        variable("bottom_face_mm", 0.6, 2.5, 1.2),
        variable("top_face_mm", 0.6, 2.5, 1.2),
    ),
    objective="mass_kg",
    constraints=(
        at_most("deflection_mm", 0.6),
        at_most("stress_mpa", 70),
        at_least("frequency_hz", 950),
    ),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
