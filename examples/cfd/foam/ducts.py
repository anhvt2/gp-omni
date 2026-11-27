"""Single-block meshes of round ducts swept along a path in the x-z plane."""

from collections.abc import Callable

import numpy as np


def disk(cells: int, wall_stretch: float = 1.5) -> np.ndarray:
    """A square lattice mapped to the unit disk, ``(cells + 1, cells + 1, 2)``.

    The outline stays a few percent short of circular at the four former corners so that
    the corner cells keep an angle below 180 degrees.
    """
    line = np.linspace(-1, 1, cells + 1)
    line = np.tanh(wall_stretch * line) / np.tanh(wall_stretch)
    u, v = np.meshgrid(line, line, indexing="ij")
    square = np.maximum(np.abs(u), np.abs(v))
    radius = np.hypot(u, v)
    factor = np.where(radius > 0, (square / np.where(radius > 0, radius, 1)) ** 0.85, 0)
    scale = factor * 2 ** (-0.0375)
    return np.stack([u * scale, v * scale], axis=-1)


def sweep(
    path: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    radius: Callable[[np.ndarray], np.ndarray],
    along: int,
    across: int,
) -> np.ndarray:
    """Lattice of a duct: ``path(s) -> (x, z)`` and ``radius(s)`` for ``s`` in [0, 1]."""
    s = np.linspace(0, 1, along + 1)
    step = 1e-6
    x, z = path(s)
    ahead, behind = np.clip(s + step, 0, 1), np.clip(s - step, 0, 1)
    dx, dz = (a - b for a, b in zip(path(ahead), path(behind)))
    length = np.hypot(dx, dz)
    dx, dz = dx / length, dz / length
    section = disk(across)
    r = radius(s)[:, None, None]
    sideways = r * section[None, :, :, 0]
    out = r * section[None, :, :, 1]
    return np.stack(
        [
            x[:, None, None] - dz[:, None, None] * sideways,
            out,
            z[:, None, None] + dx[:, None, None] * sideways,
        ],
        axis=-1,
    )


def u_bend(inlet: float, bend_radius: float, outlet: float):
    """Path along +x, turning through 180 degrees to return along -x; and its length."""
    arc = np.pi * bend_radius
    total = inlet + arc + outlet

    def path(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = np.asarray(s) * total
        angle = np.clip((distance - inlet) / bend_radius, 0, np.pi)
        x = np.where(
            distance < inlet,
            distance - inlet,
            np.where(
                distance > inlet + arc,
                -(distance - inlet - arc),
                bend_radius * np.sin(angle),
            ),
        )
        z = bend_radius * (1 - np.cos(angle))
        return x, z

    return path, total
