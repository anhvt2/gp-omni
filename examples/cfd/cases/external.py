"""External aerodynamics of bodies meshed with ``snappyHexMesh``.

All geometry is generated here, except the motorbike surface, which is read from the
OpenFOAM installation at run time.
"""

from math import cos, pi, sin

import numpy as np
from foam.case import cfd_case
from foam.shapes import loft


def _turn(triangles: np.ndarray, axis: int, angle: float, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotate triangles about a coordinate axis through ``centre``."""
    c, s = cos(angle), sin(angle)
    a, b = [(1, 2), (2, 0), (0, 1)][axis]
    shifted = triangles - np.asarray(centre)
    result = shifted.copy()
    result[..., a] = c * shifted[..., a] - s * shifted[..., b]
    result[..., b] = s * shifted[..., a] + c * shifted[..., b]
    return result + np.asarray(centre)


def _wheel(x: float, y0: float, y1: float, radius: float) -> np.ndarray:
    angle = np.linspace(0, 2 * pi, 32, endpoint=False)
    return loft(
        [
            np.column_stack(
                [x + radius * np.cos(angle), np.full(32, y), 0.95 * radius + radius * np.sin(angle)]
            )
            for y in (y0, y1)
        ]
    )


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    return cfd_case(
        name, "external", title, "snappy", solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip


_DRAG = ("drag_coefficient",)
