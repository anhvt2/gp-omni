"""Check the structured mesh writer without OpenFOAM."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/cfd"))

from foam.hex_mesh import box_grid, hex_mesh, lattice


def test_unit_box_volumes_and_closed_cells(tmp_path):
    mesh = hex_mesh()
    mesh.add_block(box_grid((0, 0, 0), (2, 1, 1), (2, 3, 4)), {"i0": "inlet", "i1": "outlet"})
    built = mesh.build()
    assert built.cell_count == 24
    np.testing.assert_allclose(built.cell_volumes(), 2 / 24)
    np.testing.assert_allclose(built.closure_error(), 0, atol=1e-12)
    assert built.patch_size("inlet") == 12
    assert built.patch_size("walls") == 2 * (2 * 3 + 2 * 4)
    built.write(tmp_path, {})
    assert (tmp_path / "constant/polyMesh/faces").is_file()


def test_adjacent_blocks_merge_into_one_domain():
    mesh = hex_mesh()
    mesh.add_block(box_grid((0, 0, 0), (1, 1, 1), (2, 2, 2)), {})
    mesh.add_block(box_grid((1, 0, 0), (2, 1, 1), (2, 2, 2)), {})
    built = mesh.build()
    assert built.cell_count == 16
    assert built.patch_size("walls") == 5 * 4 * 2


def test_separate_regions_stay_disconnected():
    mesh = hex_mesh()
    mesh.add_block(box_grid((0, 0, 0), (1, 1, 1), (2, 2, 2)), {"i1": "left_side"})
    mesh.add_block(box_grid((1, 0, 0), (2, 1, 1), (2, 2, 2)), {"i0": "right_side"}, region=1)
    built = mesh.build()
    assert built.patch_size("left_side") == built.patch_size("right_side") == 4


def test_left_handed_block_is_reoriented():
    grid = box_grid((0, 0, 0), (1, 1, 1), (2, 2, 2))[:, :, ::-1]
    mesh = hex_mesh()
    mesh.add_block(grid, {"k0": "top"})
    built = mesh.build()
    assert (built.cell_volumes() > 0).all()
    top = built.patch_points("top")
    np.testing.assert_allclose(top[:, 2], 1)


def test_collapsed_axis_edges_make_a_valid_wedge():
    x = np.linspace(0, 1, 4)
    r = np.linspace(0, 1, 3)
    angle = np.radians([-2.5, 2.5])
    grid = lattice(x, r, angle, lambda a, b, c: (a, b * np.cos(c), b * np.sin(c)))
    mesh = hex_mesh()
    mesh.add_block(grid, {"j0": "axis", "k0": "back", "k1": "front"})
    built = mesh.build()
    assert (built.cell_volumes() > 0).all()
    assert built.patch_size("axis") == 0
    np.testing.assert_allclose(built.closure_error(), 0, atol=1e-12)


def test_masked_cells_expose_a_wall():
    keep = np.ones((3, 3, 1), dtype=bool)
    keep[1, 1, 0] = False
    mesh = hex_mesh()
    mesh.add_block(
        box_grid((0, 0, 0), (3, 3, 1), (3, 3, 1)),
        {s: "outer" for s in ("i0", "i1", "j0", "j1", "k0", "k1")},
        keep=keep,
        exposed="obstacle",
    )
    assert mesh.build().patch_size("obstacle") == 4


def test_cell_zone_is_recorded(tmp_path):
    mesh = hex_mesh()
    mesh.add_block(box_grid((0, 0, 0), (1, 1, 1), (2, 1, 1)), {}, zone="rotor")
    built = mesh.build()
    built.write(tmp_path, {})
    assert "rotor" in (tmp_path / "constant/polyMesh/cellZones").read_text()


def test_inverted_cell_is_rejected():
    grid = box_grid((0, 0, 0), (1, 1, 1), (2, 1, 1))
    grid[1, :, :, 0] = 1.5  # the middle plane passes the end plane
    mesh = hex_mesh()
    mesh.add_block(grid, {})
    with pytest.raises(ValueError, match="inverted"):
        mesh.build().require_positive_volumes()
