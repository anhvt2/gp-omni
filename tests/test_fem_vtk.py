"""Check that the VTK export preserves quadratic interpolation and vector components."""

import sys
from pathlib import Path

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")
skfem = pytest.importorskip("skfem")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples/fem"), str(ROOT)]

from common import solid

from examples.postprocessing.fem import write_solution


def test_quadratic_tetrahedron_export(tmp_path):
    mesh = skfem.MeshTet()
    basis = skfem.Basis(mesh, solid.ELEMENT)
    functions = (lambda x: 2 * x[0], lambda x: -x[1], lambda x: 3 * x[2])
    displacement = np.zeros(basis.N)
    for function, dofs in zip(functions, basis.split_indices()):
        displacement[dofs] = function(basis.doflocs[:, dofs])
    result = solid.static_result(basis, displacement, solid.material(1_000, 0.25, 1.0))
    path = tmp_path / "solution.vtu"
    write_solution(path, {"load": result})

    exported = meshio.read(path)
    assert exported.cells[0].type == "tetra10"
    points = exported.points[exported.cells[0].data]
    vertices = points[:, :4]
    assert np.all(np.linalg.det(vertices[:, 1:] - vertices[:, :1]) > 0)
    for node, (a, b) in enumerate(((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)), start=4):
        np.testing.assert_allclose(points[:, node], (vertices[:, a] + vertices[:, b]) / 2)
    np.testing.assert_allclose(
        exported.point_data["load_displacement_mm"],
        np.column_stack([function(exported.points.T) for function in functions]),
    )
    stress = solid.von_mises(result)
    np.testing.assert_allclose(exported.cell_data["load_von_mises_mpa"][0], stress.values[:, 0])
