"""Read the rotational pitch of a generated, matching OpenFOAM passage."""

import math
import re
from pathlib import Path

import numpy as np

_PATCH = re.compile(r"(?m)^\s*(periodic_[ab])\s*\{([^}]+)\}")


def annulus_copies(mesh: Path) -> int:
    """Count the sectors needed to close the annulus using both cyclic face lists."""
    patches = {}
    for name, body in _PATCH.findall((mesh / "boundary").read_text()):
        patches[name] = {key: value.split() for key, value in re.findall(r"(\w+)\s+([^;]+);", body)}
    expected = {
        "type": ["cyclic"],
        "transform": ["rotational"],
        "rotationAxis": ["(0", "0", "1)"],
        "rotationCentre": ["(0", "0", "0)"],
    }
    for name, other in (("periodic_a", "periodic_b"), ("periodic_b", "periodic_a")):
        patch = patches.get(name, {})
        if any(patch.get(key) != value for key, value in expected.items()) or patch.get(
            "neighbourPatch"
        ) != [other]:
            raise ValueError(f"{mesh}: expected matching cyclic patches about +z")

    def entries(name):
        lines = (mesh / name).read_text().splitlines()
        return lines[lines.index("(") + 1 : -1]

    faces = entries("faces")
    first_vertices = []
    for name in ("periodic_a", "periodic_b"):
        patch = patches[name]
        start = int(patch["startFace"][0])
        count = int(patch["nFaces"][0])
        first_vertices.append(
            [int(face.split("(", 1)[1].split()[0]) for face in faces[start : start + count]]
        )
    if not first_vertices[0] or len(first_vertices[0]) != len(first_vertices[1]):
        raise ValueError(f"{mesh}: cyclic face counts do not match")

    points = np.fromstring(" ".join(entries("points")).replace("(", "").replace(")", ""), sep=" ")
    a, b = points.reshape(-1, 3)[np.asarray(first_vertices)]
    sample = np.argmax(np.linalg.norm(a[:, :2], axis=1))
    x, y, _ = a[sample]
    u, v, _ = b[sample]
    pitch = math.atan2(x * v - y * u, x * u + y * v)
    copies = round(math.tau / abs(pitch)) if pitch else 0
    if copies < 2 or not math.isclose(copies * abs(pitch), math.tau, abs_tol=1e-7):
        raise ValueError(f"{mesh}: cyclic pitch does not close a full annulus")
    cos, sin = math.cos(pitch), math.sin(pitch)
    rotation = np.array(((cos, -sin, 0), (sin, cos, 0), (0, 0, 1)))
    if not np.allclose(a @ rotation.T, b, rtol=0, atol=1e-8 * np.max(np.abs(points))):
        raise ValueError(f"{mesh}: cyclic faces do not match after rotation")
    return copies
