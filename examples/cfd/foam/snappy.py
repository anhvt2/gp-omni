"""Cartesian background meshes refined and snapped to surfaces with ``snappyHexMesh``."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boundaries import boundary
from .case import mesh_level
from .dictionaries import write_dictionary
from .hex_mesh import box_grid, hex_mesh
from .shapes import write_stl


@dataclass(frozen=True)
class surface:
    """Triangles forming one patch named after the surface."""

    triangles: np.ndarray
    level: tuple[int, int] = (2, 3)


@dataclass
class domain:
    """A box with a patch name per side (x0, x1, y0, y1, z0, z1) and its condition."""

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    cell: float  # background cell size at smoke resolution
    sides: Mapping[str, str]
    refine: list = field(default_factory=list)  # (lower, upper, level) boxes


def write_snappy(
    case: Path,
    box: domain,
    surfaces: Mapping[str, surface],
    patches: Mapping[str, boundary],
    inside: tuple[float, float, float],
    level: mesh_level,
) -> list[list[str]]:
    """Write the background mesh, surfaces and dictionary; return the meshing commands."""
    size = box.cell / level.resolution
    cells = [max(4, round((u - low) / size)) for low, u in zip(box.lower, box.upper)]
    keys = dict(zip(("x0", "x1", "y0", "y1", "z0", "z1"), ("i0", "i1", "j0", "j1", "k0", "k1")))
    mesh = hex_mesh()
    mesh.add_block(
        box_grid(box.lower, box.upper, cells), {keys[k]: name for k, name in box.sides.items()}
    )
    built = mesh.build()
    present = set(built.patch_names())
    built.write(case, {n: b.mesh_entry() for n, b in patches.items() if n in present})

    geometry, refinement = {}, {}
    for name, item in surfaces.items():
        write_stl(case / f"constant/triSurface/{name}.stl", {name: item.triangles})
        geometry[f"{name}.stl"] = {"type": "triSurfaceMesh", "name": name}
        kind = patches[name].mesh_entry()["type"]
        refinement[name] = {"level": item.level, "patchInfo": {"type": kind}}
    regions = {}
    for number, (lower, upper, depth) in enumerate(box.refine):
        geometry[f"region_{number}"] = {"type": "searchableBox", "min": lower, "max": upper}
        regions[f"region_{number}"] = {"mode": "inside", "levels": f"((1e15 {depth}))"}
    write_dictionary(
        case / "system/snappyHexMeshDict",
        {
            "castellatedMesh": True,
            "snap": True,
            "addLayers": False,
            "geometry": geometry,
            "castellatedMeshControls": {
                "maxLocalCells": 1000000,
                "maxGlobalCells": 600000,
                "minRefinementCells": 0,
                "nCellsBetweenLevels": 2,
                "features": (),
                "refinementSurfaces": refinement,
                "resolveFeatureAngle": 30,
                "refinementRegions": regions,
                "locationInMesh": inside,
                "allowFreeStandingZoneFaces": True,
            },
            "snapControls": {
                "nSmoothPatch": 3,
                "tolerance": 2.0,
                "nSolveIter": 50,
                "nRelaxIter": 5,
                "nFeatureSnapIter": 10,
                "implicitFeatureSnap": True,
                "explicitFeatureSnap": False,
                "multiRegionFeatureSnap": False,
            },
            "addLayersControls": {
                "relativeSizes": True,
                "layers": {},
                "expansionRatio": 1.2,
                "finalLayerThickness": 0.3,
                "minThickness": 0.1,
                "nGrow": 0,
                "featureAngle": 60,
                "nRelaxIter": 3,
                "nSmoothSurfaceNormals": 1,
                "nSmoothNormals": 3,
                "nSmoothThickness": 10,
                "maxFaceThicknessRatio": 0.5,
                "maxThicknessToMedialRatio": 0.3,
                "minMedialAxisAngle": 90,
                "nBufferCellsNoExtrude": 0,
                "nLayerIter": 50,
            },
            "meshQualityControls": {
                "maxNonOrtho": 70,
                "maxBoundarySkewness": 20,
                "maxInternalSkewness": 4,
                "maxConcave": 80,
                "minVol": 1e-15,
                "minTetQuality": 1e-15,
                "minArea": -1,
                "minTwist": 0.02,
                "minDeterminant": 0.001,
                "minFaceWeight": 0.05,
                "minVolRatio": 0.01,
                "minTriangleTwist": -1,
                "nSmoothScale": 4,
                "errorReduction": 0.75,
            },
            "mergeTolerance": 1e-6,
        },
    )
    return [["snappyHexMesh", "-overwrite"]]
