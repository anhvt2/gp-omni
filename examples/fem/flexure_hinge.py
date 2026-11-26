"""Elliptical-notch flexure hinge: compliant in rotation, stiff in the other directions.

Linear small-rotation analysis. The stress is evaluated at a design rotation of
``DESIGN_ROTATION_RAD``.

Run ``python examples/fem/flexure_hinge.py {describe,baseline,optimize}``.
"""

import numpy as np
from common import cli, solid
from common.facets import on_plane, surface_integral
from common.materials import ALUMINUM
from common.mesh import build_mesh
from common.problem import at_least, at_most, design_problem, variable
from skfem import FacetBasis

LENGTH = 40.0
MOMENT_N_MM, AXIAL_N, SIDE_N = 100.0, 100.0, 10.0
DESIGN_ROTATION_RAD = 0.01


def geometry(occ, design):
    height, neck = design["height_mm"], design["neck_mm"]
    width, half_notch = design["width_mm"], design["notch_length_mm"]
    block = [(3, occ.addBox(0, -height / 2, 0, LENGTH, height, width))]
    notches = []
    for y in (-height / 2, height / 2):
        notch = [(3, occ.addCylinder(LENGTH / 2, y, -1, 0, 0, width + 2, 1.0))]
        occ.dilate(notch, LENGTH / 2, y, -1, half_notch, (height - neck) / 2, 1)
        notches += notch
    occ.cut(block, notches)


def end_rotation(result, facets) -> float:
    """Least-squares rotation about z of the end face from its axial displacement."""
    surface = FacetBasis(result.basis.mesh, solid.ELEMENT, facets=facets, intorder=4)
    axial = np.asarray(surface.interpolate(result.displacement))[0]
    y = surface.global_coordinates()[1]
    return float(-(axial * y * surface.dx).sum() / (y**2 * surface.dx).sum())


def solve(design, mesh_scale):
    neck, half_notch = design["neck_mm"], design["notch_length_mm"]
    height, width = design["height_mm"], design["width_mm"]
    fine = (
        (LENGTH / 2 - half_notch / 2, -height / 4, 0),
        (LENGTH / 2 + half_notch / 2, height / 4, width),
    )
    mesh = build_mesh(
        lambda occ: geometry(occ, design),
        2.0 * mesh_scale,
        per_circle=24,
        refinements=[(*fine, min(max(neck / 1.5, 0.5), 1.2) * mesh_scale)],
    )
    tip = on_plane(mesh, 0, LENGTH)
    hinge = solid.structure(mesh, ALUMINUM, [solid.clamp(on_plane(mesh, 0, 0))])
    inertia = surface_integral(mesh, tip, lambda x: x[1] ** 2)

    def couple(x, normal):
        return np.stack([-MOMENT_N_MM / inertia * x[1], 0 * x[1], 0 * x[1]])

    bent = hinge.deform([solid.load(tip, couple)])
    rotation = end_rotation(bent, tip)
    stress = solid.von_mises(bent).percentile(
        99.9, keep=lambda p: np.abs(p[0] - LENGTH / 2) < half_notch
    )
    pulled = hinge.deform([solid.total_force(mesh, tip, (AXIAL_N, 0, 0))])
    sideways = hinge.deform([solid.total_force(mesh, tip, (0, 0, SIDE_N))])
    return {
        "rotational_stiffness_nm_rad": MOMENT_N_MM / rotation / 1_000,
        "stress_mpa": stress * DESIGN_ROTATION_RAD / rotation,
        "axial_stiffness_n_um": AXIAL_N / solid.mean_displacement(pulled, tip)[0] / 1_000,
        "out_of_plane_stiffness_n_mm": SIDE_N / solid.mean_displacement(sideways, tip)[2],
    }


PROBLEM = design_problem(
    name="flexure_hinge",
    title="Elliptical flexure hinge",
    variables=(
        variable("neck_mm", 0.5, 2.5, 1.2),
        variable("width_mm", 4, 15, 10),
        variable("notch_length_mm", 3, 12, 7),
        variable("height_mm", 8, 16, 12),
    ),
    objective="rotational_stiffness_nm_rad",
    constraints=(
        at_most("stress_mpa", 110),
        at_least("axial_stiffness_n_um", 50),
        at_least("out_of_plane_stiffness_n_mm", 700),
    ),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
