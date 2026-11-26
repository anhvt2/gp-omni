"""Tetrahedral meshing of Open CASCADE geometry with Gmsh.

Gmsh keeps global state: mesh one geometry at a time per process.
"""

from collections.abc import Callable, Sequence

import gmsh
import numpy as np
from skfem import MeshTet

# (lower corner, upper corner, element size) of a locally refined box
refinement = tuple[Sequence[float], Sequence[float], float]


def build_mesh(
    construct: Callable[..., None],
    size: float,
    per_circle: int = 12,
    refinements: Sequence[refinement] = (),
    boolean_tolerance: float = 0.0,
) -> MeshTet:
    """Mesh the single solid that ``construct(gmsh.model.occ)`` creates.

    ``size`` is the largest element edge and ``per_circle`` the number of elements around a
    full circle on curved surfaces. A positive ``boolean_tolerance`` (mm) makes unions of
    many intersecting surfaces fuzzy, which avoids stalls in Open CASCADE.
    """
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)  # repeatable meshes
        gmsh.option.setNumber("Geometry.ToleranceBoolean", boolean_tolerance)
        construct(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        solids = gmsh.model.getEntities(3)
        if len(solids) != 1:
            raise ValueError(f"the geometry must be one solid, not {len(solids)}")
        smallest = min([size, *(box[2] for box in refinements)])
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", smallest / 3)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", per_circle)
        _refine(refinements, size)
        gmsh.model.mesh.generate(3)
        return _extract()
    finally:
        gmsh.finalize()


def prism(occ, corners: Sequence[Sequence[float]], direction: Sequence[float]) -> list:
    """Extrude the planar polygon ``corners`` along ``direction``; return its volume."""
    points = [occ.addPoint(*corner) for corner in corners]
    lines = [occ.addLine(start, end) for start, end in zip(points, points[1:] + points[:1])]
    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    return [part for part in occ.extrude([(2, face)], *direction) if part[0] == 3]


def _refine(refinements: Sequence[refinement], size: float) -> None:
    fields = []
    for lower, upper, inner in refinements:
        field = gmsh.model.mesh.field.add("Box")
        settings = {"VIn": inner, "VOut": size, "Thickness": 2 * size}
        settings |= dict(zip(("XMin", "YMin", "ZMin"), lower))
        settings |= dict(zip(("XMax", "YMax", "ZMax"), upper))
        for key, value in settings.items():
            gmsh.model.mesh.field.setNumber(field, key, value)
        fields.append(field)
    if fields:
        combined = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(combined, "FieldsList", fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(combined)


def _extract() -> MeshTet:
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    _, corners = gmsh.model.mesh.getElementsByType(4)  # linear tetrahedra
    if corners.size == 0:
        raise ValueError("Gmsh produced no tetrahedra")
    row = np.zeros(int(tags.max()) + 1, int)
    row[tags.astype(int)] = np.arange(tags.size)
    elements = row[corners.astype(int)].reshape(-1, 4)
    used, elements = np.unique(elements, return_inverse=True)
    points = coordinates.reshape(-1, 3)[used]
    return MeshTet(np.ascontiguousarray(points.T), np.ascontiguousarray(elements.reshape(-1, 4).T))
