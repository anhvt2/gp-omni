"""Geometry and physics checks of the 3D design cases in ``examples/fem``."""

import subprocess
import sys
from functools import cache
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skfem")
pytest.importorskip("gmsh")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/fem"
sys.path.insert(0, str(EXAMPLES))

from common.cases import CASES
from common.mesh import build_mesh, prism
from common.solid import volume_mm3

PROBLEMS = {name: import_module(name).PROBLEM for name in CASES}
COARSE = 1.5  # element size factor that keeps the tests fast


@cache
def analyze(name, **changes):
    problem = PROBLEMS[name]
    return problem.analyze({**problem.reference_design(), **changes}, COARSE)


def test_every_case_is_one_file_directly_in_the_fem_folder():
    assert len(CASES) == 10
    for name, problem in PROBLEMS.items():
        assert problem.source == EXAMPLES / f"{name}.py"


@pytest.mark.parametrize("name", CASES)
def test_each_case_runs_as_its_own_script(name):
    listing = subprocess.run(
        [sys.executable, EXAMPLES / f"{name}.py", "describe"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert PROBLEMS[name].objective in listing
    assert all(v.name in listing for v in PROBLEMS[name].variables)


def test_prism_extrudes_a_polygon():
    def wedge(occ):
        prism(occ, [(0, 0, 0), (10, 0, 0), (0, 0, 6)], (0, 4, 0))

    assert volume_mm3(build_mesh(wedge, 2.0)) == pytest.approx(10 * 6 / 2 * 4, rel=1e-6)


def test_build_mesh_matches_the_geometry_volume():
    def block_with_hole(occ):
        block = occ.addBox(0, 0, 0, 40, 20, 10)
        hole = occ.addCylinder(20, 10, -1, 0, 0, 12, 5)
        occ.cut([(3, block)], [(3, hole)])

    mesh = build_mesh(block_with_hole, 2.0)
    assert volume_mm3(mesh) == pytest.approx(40 * 20 * 10 - np.pi * 25 * 10, rel=0.01)


def test_build_mesh_rejects_disconnected_geometry():
    def two_blocks(occ):
        occ.addBox(0, 0, 0, 1, 1, 1)
        occ.addBox(5, 0, 0, 1, 1, 1)

    with pytest.raises(ValueError, match="one solid"):
        build_mesh(two_blocks, 0.5)


@pytest.mark.parametrize("name", CASES)
def test_variables_and_constraints_are_well_formed(name):
    problem = PROBLEMS[name]
    assert problem.name == name
    assert 3 <= len(problem.variables) <= 6
    assert all(v.lower <= v.reference <= v.upper for v in problem.variables)
    assert problem.constraints


@pytest.mark.parametrize("name", CASES)
def test_reference_design_is_feasible(name):
    problem = PROBLEMS[name]
    metrics = analyze(name)
    assert all(np.isfinite(value) and value > 0 for value in metrics.values()), metrics
    assert problem.is_feasible(metrics), metrics


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("corner", [0.0, 1.0])
def test_bound_corners_give_valid_geometry(name, corner):
    problem = PROBLEMS[name]
    design = problem.design_from_unit(np.full(len(problem.variables), corner))
    metrics = problem.analyze(design, COARSE)
    assert all(np.isfinite(value) and value > 0 for value in metrics.values()), metrics


def test_design_outside_the_bounds_is_rejected():
    problem = PROBLEMS["drone_arm"]
    with pytest.raises(ValueError, match="wall_mm"):
        problem.analyze({**problem.reference_design(), "wall_mm": 100.0}, COARSE)


@pytest.mark.parametrize(
    "name, variable, value, metric, direction",
    [
        ("drone_arm", "wall_mm", 2.2, "deflection_mm", -1),
        ("drone_arm", "height_mm", 20.0, "frequency_hz", 1),
        ("l_bracket", "gusset_size_mm", 55.0, "deflection_mm", -1),
        ("heat_sink", "pin_height_mm", 30.0, "temperature_rise_k", -1),
        ("stiffened_panel", "rib_height_mm", 20.0, "frequency_hz", 1),
        ("connecting_rod", "depth_mm", 20.0, "lateral_deflection_mm", -1),
        ("lifting_lug", "thickness_mm", 22.0, "stress_mpa", -1),
        ("pressure_nozzle", "shell_wall_mm", 12.0, "stress_mpa", -1),
        ("flexure_hinge", "neck_mm", 1.8, "rotational_stiffness_nm_rad", 1),
        ("cold_plate", "channel_diameter_mm", 4.0, "pressure_drop_kpa", 1),
        ("lattice_beam", "core_height_mm", 14.0, "deflection_mm", -1),
    ],
)
def test_design_changes_move_the_response_as_physics_requires(
    name, variable, value, metric, direction
):
    reference = analyze(name)[metric]
    changed = analyze(name, **{variable: value})[metric]
    assert direction * (changed - reference) > 0


def test_heat_sink_rejects_the_input_power_by_convection():
    assert analyze("heat_sink")["convected_w"] == pytest.approx(15.0, rel=1e-4)


def test_cold_plate_rejects_the_chip_power_into_the_coolant():
    assert analyze("cold_plate")["convected_w"] == pytest.approx(90.0, rel=1e-4)


def test_cold_plate_pressure_drop_matches_blasius():
    diameter, flow = 0.005, 1.5e-3 / 60
    velocity = flow / (np.pi * diameter**2 / 4)
    reynolds = 997 * velocity * diameter / 0.00089
    friction = 0.316 * reynolds**-0.25
    expected = friction * 0.1 / diameter * 997 * velocity**2 / 2 / 1_000
    drop = analyze("cold_plate", channel_diameter_mm=5.0)["pressure_drop_kpa"]
    assert drop == pytest.approx(expected, rel=1e-6)


def test_nozzle_shell_stress_matches_thin_cylinder_theory():
    reference = PROBLEMS["pressure_nozzle"].reference_design()
    wall = reference["shell_wall_mm"]
    mean_radius = 150 + wall / 2
    # closed cylinder: hoop = p r / t, axial = hoop / 2, radial ~ 0
    expected = np.sqrt(3) / 2 * 1.5 * mean_radius / wall
    assert analyze("pressure_nozzle")["shell_stress_mpa"] == pytest.approx(expected, rel=0.1)


def test_flexure_stiffness_is_close_to_beam_theory():
    design = PROBLEMS["flexure_hinge"].reference_design()
    height, neck = design["height_mm"], design["neck_mm"]
    half_notch, width = design["notch_length_mm"], design["width_mm"]
    x = np.linspace(0, 40, 40_001)
    offset = np.clip((x - 20) / half_notch, -1, 1)
    thickness = height - (height - neck) * np.sqrt(1 - offset**2)
    compliance = np.trapezoid(12 / (69_000 * width * thickness**3), x)  # rad per N mm
    stiffness = analyze("flexure_hinge")["rotational_stiffness_nm_rad"]
    assert stiffness == pytest.approx(1 / compliance / 1_000, rel=0.25)
