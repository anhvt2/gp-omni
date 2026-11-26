"""Boundary facet selection and surface integrals on tetrahedral meshes."""

from collections.abc import Callable, Sequence

import numpy as np
from skfem import ElementTetP1, FacetBasis, Functional, MeshTet

TOLERANCE_MM = 1e-3  # mesh nodes lie on the CAD surfaces
point_predicate = Callable[[np.ndarray], np.ndarray]  # points (3, n) -> booleans (n,)


def select_facets(mesh: MeshTet, predicate: point_predicate) -> np.ndarray:
    """Boundary facets whose three vertices all satisfy ``predicate``."""
    boundary = mesh.boundary_facets()
    inside = np.asarray(predicate(mesh.p), bool)
    facets = boundary[inside[mesh.facets[:, boundary]].all(axis=0)]
    if facets.size == 0:
        raise ValueError("the predicate selected no boundary facet")
    return facets


def on_plane(mesh: MeshTet, axis: int, value: float) -> np.ndarray:
    """Boundary facets in the plane ``x[axis] == value``."""
    return select_facets(mesh, lambda p: np.abs(p[axis] - value) < TOLERANCE_MM)


def on_cylinder(
    mesh: MeshTet,
    axis: int,
    center: Sequence[float],
    radius: float,
    keep: point_predicate | None = None,
) -> np.ndarray:
    """Boundary facets on the cylinder of ``radius`` whose axis runs along ``axis``.

    ``center`` is any point of that axis; ``keep(points)`` restricts the surface further.
    """
    first, second = (i for i in range(3) if i != axis)

    def surface(p):
        distance = np.hypot(p[first] - center[first], p[second] - center[second])
        found = np.abs(distance - radius) < TOLERANCE_MM
        return found if keep is None else found & keep(p)

    return select_facets(mesh, surface)


def surface_integral(
    mesh: MeshTet, facets: np.ndarray, integrand: Callable[[np.ndarray], np.ndarray]
) -> float:
    """Integral of ``integrand(x)`` over the facets."""
    basis = FacetBasis(mesh, ElementTetP1(), facets=facets, intorder=6)
    return float(Functional(lambda w: integrand(w.x) * np.ones_like(w.x[0])).assemble(basis))


def area(mesh: MeshTet, facets: np.ndarray) -> float:
    return surface_integral(mesh, facets, lambda x: 1.0)
