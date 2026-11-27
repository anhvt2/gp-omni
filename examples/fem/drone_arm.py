"""Hollow tapered drone arm with lightening holes, clamped at the root.

The motor is a lumped mass spread over the tip face. Deflection is checked under a
maneuver thrust and stress under a static sideways impact load at the tip.

Run ``python examples/fem/drone_arm.py {describe,baseline,optimize}``.
"""

from common import cli, solid
from common.facets import on_plane
from common.materials import ALUMINUM
from common.mesh import build_mesh, prism
from common.problem import at_least, at_most, design_problem, variable

LENGTH = 180.0
ROOT_PLUG, TIP_PLUG = 5.0, 14.0
HOLE_CENTERS = (35.0, 70.0, 105.0, 140.0)
MOTOR_MASS_T = 6e-5
THRUST_N = 24.0
SIDE_IMPACT_N = 150.0


def geometry(occ, design):
    root, tip = design["root_width_mm"], design["tip_width_mm"]
    height, wall = design["height_mm"], design["wall_mm"]

    def tapered(start, end, shrink, z, depth):
        half = [(root + (tip - root) * x / LENGTH) / 2 - shrink for x in (start, end)]
        plan = [(start, -half[0]), (end, -half[1]), (end, half[1]), (start, half[0])]
        return prism(occ, [(x, y, z) for x, y in plan], (0, 0, depth))

    outer = tapered(0, LENGTH, 0, -height / 2, height)
    voids = tapered(ROOT_PLUG, LENGTH - TIP_PLUG, wall, -height / 2 + wall, height - 2 * wall)
    radius = design["hole_ratio"] * height
    voids += [(3, occ.addCylinder(x, -30, 0, 0, 60, 0, radius)) for x in HOLE_CENTERS]
    occ.cut(outer, voids)


def prepare_mesh(design, mesh_scale):
    return build_mesh(lambda occ: geometry(occ, design), 3.5 * mesh_scale)


def solve(design, mesh_scale):
    return solve_mesh(design, prepare_mesh(design, mesh_scale))


def solve_mesh(design, mesh, fields=None):
    tip = on_plane(mesh, 0, LENGTH)
    thrust = solid.total_force(mesh, tip, (0, 0, THRUST_N))
    arm = solid.structure(mesh, ALUMINUM, [solid.clamp(on_plane(mesh, 0, 0))])
    lifted = arm.deform([thrust])
    struck = arm.deform([solid.total_force(mesh, tip, (0, SIDE_IMPACT_N, 0))])
    frequency = arm.frequencies(point_masses=[(tip, MOTOR_MASS_T)])[0]
    if fields is not None:
        fields.update(thrust=lifted, side_impact=struck)
    return {
        "mass_kg": solid.mass_kg(mesh, ALUMINUM),
        "deflection_mm": float(solid.mean_displacement(lifted, tip)[2]),
        "stress_mpa": solid.von_mises(struck).percentile(99, keep=lambda p: p[0] > 8),
        "frequency_hz": float(frequency),
    }


PROBLEM = design_problem(
    name="drone_arm",
    title="Hollow tapered drone motor arm",
    variables=(
        variable("root_width_mm", 16, 30, 24),
        variable("tip_width_mm", 14, 22, 18),
        variable("height_mm", 10, 22, 16),
        variable("wall_mm", 0.8, 2.5, 1.5),
        variable("hole_ratio", 0.1, 0.38, 0.2),
    ),
    objective="mass_kg",
    constraints=(
        at_most("deflection_mm", 0.22),
        at_most("stress_mpa", 50),
        at_least("frequency_hz", 200),
    ),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
