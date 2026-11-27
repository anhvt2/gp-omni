"""Solver run time on the motorbike geometry."""

import gzip
import os
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
from foam.case import at_most, variable
from foam.snappy import surface

from ..external import _DRAG, _case
from ..external_flow import wind_tunnel


def _read_obj(path: Path) -> np.ndarray:
    """All polygons of a gzipped Wavefront file, fan triangulated."""
    vertices, triangles = [], []
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                ids = [int(token.split("/")[0]) for token in line.split()[1:]]
                ids = [i - 1 if i > 0 else len(vertices) + i for i in ids]
                triangles += [(ids[0], b, c) for b, c in pairwise(ids[1:])]
    return np.asarray(vertices)[np.asarray(triangles)]


def motorbike_benchmark(case, design, level):
    """The OpenFOAM motorbike geometry; the objective is the solver's run time."""
    root = os.environ.get("FOAM_TUTORIALS")
    if not root:
        raise RuntimeError("FOAM_TUTORIALS is unset; run inside the OpenFOAM container")
    bike = _read_obj(Path(root) / "resources/geometry/motorBike.obj.gz")
    depth = round(design["surface_level"])

    def run_time(directory, metrics):
        found = re.findall(
            r"ExecutionTime = ([\d.]+) s", (directory / "log.simpleFoam").read_text()
        )
        return {"execution_time_s": float(found[-1])}

    return wind_tunnel(
        case, level, {"motorbike": surface(bike, (depth, depth))},
        (-3.0, -3.0, 0.0), (7.0, 3.0, 4.0), design["background_cell_m"],
        speed=20.0, length=1.42, area=0.75, derive=run_time,
    )  # fmt: skip


CASE = _case(
    "motorbike_benchmark",
    "Solver run time on the motorbike geometry",
    motorbike_benchmark,
    (variable("surface_level", 3.0, 5.0, 4.0), variable("background_cell_m", 0.3, 0.6, 0.4)),
    "execution_time_s",
    (at_most("drag_coefficient", 0.25),),
    positive=_DRAG,
    simplification="speed benchmark; processes are set with --cores",
)
