"""Export native quadratic tetrahedral fields to VTK XML."""

import meshio
import numpy as np

VTK_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def vtk_tetrahedra(basis):
    local = basis.elem.doflocs
    mids = [(local[a] + local[b]) / 2 for a, b in VTK_EDGES]
    order = [0, 1, 2, 3, *(np.flatnonzero(np.all(local == mid, axis=1)).item() for mid in mids)]
    cells = basis.element_dofs[np.asarray(order)].T.copy()
    corners = basis.doflocs.T[cells[:, :4]]
    flipped = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0
    # Reverse vertices 0 and 1 and remap the six VTK edge nodes.
    cells[flipped] = cells[flipped][:, [1, 0, 2, 3, 4, 6, 5, 8, 7, 9]]
    return cells


def write_solution(path, fields):
    from common.solid import static_result, von_mises

    point_data, cell_data = {}, {}
    for name, field in fields.items():
        if isinstance(field, static_result):
            components, bases = zip(*field.basis.split(field.displacement))
            basis = bases[0]
            point_data[f"{name}_displacement_mm"] = np.column_stack(components)
            stress = von_mises(field)
            mean = (stress.values * stress.weights).sum(axis=1) / stress.weights.sum(axis=1)
            cell_data[f"{name}_von_mises_mpa"] = [mean]
        else:
            basis = field.basis
            point_data[f"{name}_k"] = field.values

    meshio.write_points_cells(
        path,
        basis.doflocs.T,
        [("tetra10", vtk_tetrahedra(basis))],
        point_data=point_data,
        cell_data=cell_data,
    )
    return {"point": list(point_data), "cell": list(cell_data)}
