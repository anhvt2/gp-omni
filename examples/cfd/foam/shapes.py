"""Closed triangle surfaces of simple bodies, shape ``(n, 3, 3)``, and STL output."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def loft(sections) -> np.ndarray:
    """Join closed loops of equal size into a tube with fan-triangulated end caps."""
    rings = np.asarray(sections, dtype=float)
    if rings.ndim != 3 or rings.shape[0] < 2 or rings.shape[1] < 3:
        raise ValueError("a loft needs at least two loops of at least three points")
    a = rings[:-1]
    b = np.roll(rings[:-1], -1, axis=1)
    c = np.roll(rings[1:], -1, axis=1)
    d = rings[1:]
    sides = np.concatenate([np.stack([a, b, c], axis=2), np.stack([a, c, d], axis=2)])
    triangles = [sides.reshape(-1, 3, 3)]
    for ring, reverse in ((rings[0], True), (rings[-1], False)):
        centre = np.broadcast_to(ring.mean(axis=0), ring.shape)
        following = np.roll(ring, -1, axis=0)
        corners = (centre, following, ring) if reverse else (centre, ring, following)
        triangles.append(np.stack(corners, axis=1))
    result = np.concatenate(triangles)
    area = np.linalg.norm(
        np.cross(result[:, 1] - result[:, 0], result[:, 2] - result[:, 0]), axis=1
    )
    result = result[area > 1e-14]
    return result if volume(result) >= 0 else result[:, ::-1]


def volume(triangles: np.ndarray) -> float:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6)


def prism(profile_xz, y0: float, y1: float) -> np.ndarray:
    """Extrude a convex ``(x, z)`` outline along y."""
    outline = np.asarray(profile_xz, dtype=float)
    return loft(
        [np.column_stack([outline[:, 0], np.full(len(outline), y), outline[:, 1]])
         for y in (y0, y1)]
    )  # fmt: skip


def box(lower, upper) -> np.ndarray:
    outline = [
        (lower[0], lower[2]),
        (upper[0], lower[2]),
        (upper[0], upper[2]),
        (lower[0], upper[2]),
    ]
    return prism(outline, lower[1], upper[1])


def revolve(x, radius, segments: int = 32, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """A body of revolution about an axis parallel to x."""
    angle = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = [
        np.column_stack(
            [np.full(segments, xi), max(r, 1e-6) * np.cos(angle), max(r, 1e-6) * np.sin(angle)]
        )
        + np.asarray(centre)
        for xi, r in zip(x, radius)
    ]
    return loft(rings)


def naca_outline(chord: float, thickness: float = 0.12, camber: float = 0.0, points: int = 30):
    """Closed ``(x, z)`` loop of a four-digit style section with a sharp trailing edge."""
    beta = np.linspace(0, np.pi, points)
    x = 0.5 * (1 - np.cos(beta))
    half = 5 * thickness * (
        0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    line = 4 * camber * x * (1 - x)
    upper = np.column_stack([x, line + half])
    lower = np.column_stack([x, line - half])
    return chord * np.vstack([upper, lower[-2:0:-1]])


def wing(stations, thickness: float = 0.12, camber: float = 0.02) -> np.ndarray:
    """Loft sections given as ``(y, x_leading, z, chord, twist_degrees)``."""
    rings = []
    for y, x0, z0, chord, twist in stations:
        outline = naca_outline(chord, thickness, camber)
        angle = np.radians(twist)  # positive raises the leading edge
        x = outline[:, 0] - 0.25 * chord
        z = outline[:, 1]
        rings.append(
            np.column_stack(
                [
                    x0 + 0.25 * chord + x * np.cos(angle) + z * np.sin(angle),
                    np.full(len(outline), y),
                    z0 - x * np.sin(angle) + z * np.cos(angle),
                ]
            )
        )
    return loft(rings)


def sphere(radius: float, centre=(0.0, 0.0, 0.0), segments: int = 64, relief=None) -> np.ndarray:
    """A sphere; ``relief(direction) -> depth`` shortens the radius locally."""
    latitude = np.linspace(-np.pi / 2, np.pi / 2, segments // 2 + 1)
    longitude = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = []
    for phi in latitude:
        inner = max(np.cos(phi), 1e-4)
        direction = np.column_stack(
            [
                inner * np.cos(longitude),
                inner * np.sin(longitude),
                np.full(segments, np.sin(phi)),
            ]
        )
        depth = 0.0 if relief is None else relief(direction)[:, None]
        rings.append(np.asarray(centre) + direction * (radius - depth))
    return loft(rings)


def write_stl(path: Path, solids: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for name, triangles in solids.items():
            stream.write(f"solid {name}\n")
            for a, b, c in triangles:
                normal = np.cross(b - a, c - a)
                normal /= max(np.linalg.norm(normal), 1e-30)
                stream.write(
                    "facet normal {:.7g} {:.7g} {:.7g}\n outer loop\n".format(*normal)
                    + "".join("  vertex {:.9g} {:.9g} {:.9g}\n".format(*p) for p in (a, b, c))
                    + " endloop\nendfacet\n"
                )
            stream.write(f"endsolid {name}\n")
