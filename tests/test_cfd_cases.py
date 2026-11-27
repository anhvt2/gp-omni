"""Every CFD case builds its mesh and dictionaries without running a solver."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gmsh")
foamlib = pytest.importorskip("foamlib")
sys.path.insert(0, str(Path(__file__).parents[1] / "examples/cfd"))

from cases import CASES
from foam.case import SMOKE
from foam.passage import axial_channel, blade_row, build_mesh, machine

TURBOMACHINERY = 23
NEEDS_INSTALLATION = ("motorbike_benchmark",)


def test_periodic_faces_match_in_order_after_rotation():
    row = blade_row("rotor", 0.35, 0.6, (-50.0, -65.0), (-40.0, -58.0), omega=100.0)
    wheel = machine(*axial_channel(0.3, 0.125, 0.25), blades=9, rows=(row,))
    built = build_mesh(wheel, SMOKE).build()
    patches = built._boundary()
    turn = 2 * np.pi / wheel.blades
    rotation = np.array(
        [[np.cos(turn), -np.sin(turn), 0], [np.sin(turn), np.cos(turn), 0], [0, 0, 1]]
    )
    pairs = zip(patches["periodic_a"], patches["periodic_b"], strict=True)
    for first, second in pairs:
        a = built.points[built.faces[first]] @ rotation.T
        b = built.points[built.faces[second]]
        # the same first point, then the opposite sense
        np.testing.assert_allclose(a[[0, *range(len(a) - 1, 0, -1)]], b, atol=1e-9)


def test_rushton_blades_stay_inside_the_impeller(tmp_path):
    case = CASES["rushton_impeller"]
    design = case.reference_design()
    case.build(tmp_path, design, SMOKE)
    mesh = tmp_path / "constant/polyMesh"
    points = np.array(
        [line.strip("()").split() for line in _entries(mesh / "points")], dtype=float
    )
    boundary = foamlib.FoamFile(mesh / "boundary").as_dict()
    patches = next(v for v in boundary.values() if isinstance(v, list))
    impeller = dict(patches)["impeller"]
    start, count = int(impeller["startFace"]), int(impeller["nFaces"])
    faces = _entries(mesh / "faces")[start : start + count]
    used = {int(p) for face in faces for p in face[face.index("(") + 1 : -1].split()}
    reach = np.hypot(*points[sorted(used), :2].T).max()
    assert reach <= design["impeller_diameter_m"] / 2 + 1e-9


def _entries(path: Path) -> list[str]:
    """The lines of a list file written by the mesh writer."""
    lines = path.read_text().splitlines()
    start = lines.index("(")
    return lines[start + 1 : -1]


def test_registry_covers_the_case_studies():
    assert len(CASES) == 49
    families = [c.family for c in CASES.values()]
    assert families.count("axial_turbo") + families.count("radial_turbo") == TURBOMACHINERY
    assert families.count("internal") == 13
    assert families.count("external") == 13


@pytest.mark.parametrize("name", sorted(CASES))
def test_definition_is_consistent(name):
    case = CASES[name]
    case.check_design(case.reference_design())
    assert all(v.lower < v.upper for v in case.variables)
    assert case.constraints
    assert case.simplification or case.mesher


@pytest.mark.parametrize("corner", ["reference", "lower", "upper"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_case_builds(name, corner, tmp_path):
    if name in NEEDS_INSTALLATION and "FOAM_TUTORIALS" not in os.environ:
        pytest.skip("needs the OpenFOAM installation")
    case = CASES[name]
    design = (
        case.reference_design()
        if corner == "reference"
        else {v.name: getattr(v, corner) for v in case.variables}
    )
    plan = case.build(tmp_path, design, SMOKE)
    assert plan.solve
    control = (tmp_path / "system/controlDict").read_text()
    assert case.solver in control
    assert (tmp_path / "0/U").is_file()
    if case.mesher == "snappy":
        assert plan.mesh[0][0] == "snappyHexMesh"
    if corner == "reference" and case.mesher != "bundled":
        # every generated dictionary parses
        for path in [*(tmp_path / "system").iterdir(), *(tmp_path / "0").iterdir()]:
            foamlib.FoamFile(path).as_dict()
