"""Watertight patch surfaces of internal-flow geometries, built with Gmsh's OpenCASCADE.

Gmsh keeps global state, so call this from one thread; the command line runs one case per
process.
"""

from collections.abc import Callable

import numpy as np


def fluid_surfaces(
    build: Callable,
    classify: Callable[[np.ndarray], str],
    size: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Triangles of every patch of the fluid region, and the fluid volume.

    ``build(occ)`` creates the geometry and returns the ``(3, tag)`` volumes of the fluid.
    ``classify`` names a face from its centre of mass.
    """
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        volumes = build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        if len(volumes) != 1:
            raise ValueError(f"the fluid must be one volume, got {len(volumes)}")
        faces = gmsh.model.getBoundary(volumes, oriented=False)
        total = gmsh.model.occ.getMass(*volumes[0])
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        position = np.zeros((int(tags.max()) + 1, 3))
        position[tags.astype(int)] = coordinates.reshape(-1, 3)
        patches: dict[str, list] = {}
        for dimension, tag in faces:
            name = classify(np.array(gmsh.model.occ.getCenterOfMass(dimension, tag)))
            kinds, _, nodes = gmsh.model.mesh.getElements(dimension, tag)
            for kind, ids in zip(kinds, nodes):
                if kind == 2:
                    patches.setdefault(name, []).append(position[ids.astype(int)].reshape(-1, 3, 3))
        return {name: np.concatenate(parts) for name, parts in patches.items()}, float(total)
    finally:
        gmsh.finalize()


def tube(occ, start, end, radius: float, end_radius: float | None = None) -> tuple[int, int]:
    """A cylinder, or a cone if ``end_radius`` differs, from ``start`` to ``end``."""
    start, axis = np.asarray(start, float), np.subtract(end, start, dtype=float)
    if end_radius is None or abs(end_radius - radius) < 1e-12:
        return 3, occ.addCylinder(*start, *axis, radius)
    return 3, occ.addCone(*start, *axis, radius, end_radius)


def fuse(occ, volumes: list) -> list:
    """The union of overlapping volumes."""
    if len(volumes) == 1:
        return volumes
    result, _ = occ.fuse(volumes[:1], volumes[1:])
    return [v for v in result if v[0] == 3]


def near(point, tolerance: float = 1e-6) -> Callable[[np.ndarray], bool]:
    """Whether a face centre lies at ``point``, to recognise end caps."""
    return lambda centre: bool(np.linalg.norm(centre - np.asarray(point)) < tolerance)


def name_caps(caps: dict, other: str = "walls") -> Callable[[np.ndarray], str]:
    """Classify faces by the centres of the openings in ``caps``."""

    def classify(centre: np.ndarray) -> str:
        for name, point in caps.items():
            if np.linalg.norm(centre - np.asarray(point)) < 1e-6:
                return name
        return other

    return classify
