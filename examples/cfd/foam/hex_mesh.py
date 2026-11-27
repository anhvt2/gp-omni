"""Write structured hexahedral blocks as one OpenFOAM ``polyMesh``.

Blocks are point lattices of shape ``(ni + 1, nj + 1, nk + 1, 3)``. Coincident points are
merged, so matching blocks join conformally; blocks in different ``region`` numbers stay
disconnected, which is what sliding interfaces need. Collapsed edges (an axis of rotation)
are allowed. Faces with a single owner become boundary faces named by the block side.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SIDES = ("i0", "i1", "j0", "j1", "k0", "k1")
# Corner offsets of a hexahedron, and each face listed so that its normal points outward.
CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
FACES = ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7))


def lattice(u: np.ndarray, v: np.ndarray, w: np.ndarray, mapping: Callable) -> np.ndarray:
    """Points of ``mapping(u, v, w) -> (x, y, z)`` on the tensor product of three axes."""
    a, b, c = np.meshgrid(u, v, w, indexing="ij")
    return np.stack(np.broadcast_arrays(*mapping(a, b, c)), axis=-1).astype(float)


def box_grid(lower, upper, cells) -> np.ndarray:
    """Uniform lattice of an axis-aligned box."""
    axes = [np.linspace(lower[d], upper[d], cells[d] + 1) for d in range(3)]
    return lattice(*axes, lambda a, b, c: (a, b, c))


@dataclass
class _block:
    points: np.ndarray
    names: list  # six arrays of patch names, one per cell face of that side
    keep: np.ndarray
    exposed: str
    zone: str | None
    region: int


class hex_mesh:
    """Collect blocks, then :meth:`build` the connected mesh."""

    def __init__(self):
        self.blocks: list[_block] = []

    def add_block(
        self,
        points: np.ndarray,
        sides: Mapping[str, object],
        zone: str | None = None,
        region: int = 0,
        keep: np.ndarray | None = None,
        exposed: str = "walls",
    ) -> None:
        """Add a lattice. ``sides`` maps a side to a patch name or an array of names over
        that side; unnamed sides become ``exposed``, as do faces next to removed cells.
        ``zone`` may also be an object array of cell shape holding zone names or ``None``.
        """
        points = np.asarray(points, dtype=float)
        cells = tuple(n - 1 for n in points.shape[:3])
        if len(points.shape) != 4 or points.shape[3] != 3 or min(cells) < 1:
            raise ValueError("a block needs shape (ni+1, nj+1, nk+1, 3)")
        unknown = set(sides) - set(SIDES)
        if unknown:
            raise ValueError(f"unknown block sides {sorted(unknown)}")
        keep = np.ones(cells, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
        names = []
        for index, side in enumerate(SIDES):
            shape = tuple(c for d, c in enumerate(cells) if d != index // 2)
            names.append(np.broadcast_to(np.asarray(sides.get(side, exposed), object), shape))
        if _signed_volume(points) < 0:  # left-handed lattice: reverse the third axis
            points, keep = points[:, :, ::-1], keep[:, :, ::-1]
            if isinstance(zone, np.ndarray):
                zone = zone[:, :, ::-1]
            names[:4] = [n[:, ::-1] for n in names[:4]]
            names[4], names[5] = names[5], names[4]
        self.blocks.append(_block(points, names, keep, exposed, zone, region))

    def build(self, tolerance: float | None = None) -> "built_mesh":
        if not self.blocks:
            raise ValueError("the mesh has no blocks")
        raw = np.concatenate([b.points.reshape(-1, 3) for b in self.blocks])
        regions = np.concatenate(
            [np.full(b.points[..., 0].size, b.region) for b in self.blocks]
        )
        if tolerance is None:
            tolerance = 1e-8 * float(np.ptp(raw, axis=0).max())
        merged = _merge(raw, regions, tolerance)
        cells, names, zones, offset = [], [], [], 0
        for block in self.blocks:
            shape = block.points.shape[:3]
            ids = merged[offset : offset + np.prod(shape)].reshape(shape)
            offset += np.prod(shape)
            ni, nj, nk = (n - 1 for n in shape)
            corners = np.stack(
                [ids[a : a + ni, b : b + nj, c : c + nk] for a, b, c in CORNERS], axis=-1
            )
            tags = np.full((ni, nj, nk, 6), block.exposed, dtype=object)
            tags[0, :, :, 0], tags[-1, :, :, 1] = block.names[0], block.names[1]
            tags[:, 0, :, 2], tags[:, -1, :, 3] = block.names[2], block.names[3]
            tags[:, :, 0, 4], tags[:, :, -1, 5] = block.names[4], block.names[5]
            zone = block.zone
            zone = zone if isinstance(zone, np.ndarray) else np.full((ni, nj, nk), zone, object)
            cells.append(corners[block.keep])
            names.append(tags[block.keep])
            zones.append(zone[block.keep])
        first = np.unique(merged, return_index=True)[1]
        return _connect(raw[first], np.concatenate(cells), np.concatenate(names),
                        np.concatenate(zones))  # fmt: skip


def _signed_volume(points: np.ndarray) -> float:
    """Summed triple product of the cell edges; negative for a left-handed lattice."""
    p = points
    a = p[1:, :-1, :-1] - p[:-1, :-1, :-1]
    b = p[:-1, 1:, :-1] - p[:-1, :-1, :-1]
    c = p[:-1, :-1, 1:] - p[:-1, :-1, :-1]
    d = p[1:, 1:, 1:] - p[:-1, 1:, 1:]
    e = p[1:, 1:, 1:] - p[1:, :-1, 1:]
    f = p[1:, 1:, 1:] - p[1:, 1:, :-1]
    low = np.einsum("...i,...i", np.cross(a, b), c)
    high = np.einsum("...i,...i", np.cross(d, e), f)
    return float(low.sum() + high.sum())


def _merge(points: np.ndarray, regions: np.ndarray, tolerance: float) -> np.ndarray:
    """New consecutive point number of every raw point."""
    pairs = cKDTree(points).query_pairs(tolerance, output_type="ndarray")
    pairs = pairs[regions[pairs[:, 0]] == regions[pairs[:, 1]]]
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2
    )
    return connected_components(graph, directed=False)[1]


def _connect(points, cells, names, zones) -> "built_mesh":
    faces: dict[tuple, int] = {}
    loops, owner, neighbour, tags = [], [], [], []
    for cell, corners in enumerate(cells.tolist()):
        for side, local in enumerate(FACES):
            loop = [corners[v] for v in local]
            loop = [p for n, p in enumerate(loop) if p != loop[n - 1]]
            if len(set(loop)) < 3:
                continue  # the face collapsed onto an edge
            key = tuple(sorted(set(loop)))
            known = faces.get(key)
            if known is None:
                faces[key] = len(loops)
                loops.append(loop)
                owner.append(cell)
                neighbour.append(-1)
                tags.append(names[cell, side])
            elif neighbour[known] == -1:
                neighbour[known] = cell
            else:
                raise ValueError("more than two cells share a face")
    used = np.zeros(len(points), dtype=bool)
    for loop in loops:
        used[loop] = True
    renumber = np.cumsum(used) - 1
    loops = [renumber[loop].tolist() for loop in loops]
    return built_mesh(
        points[used], loops, np.array(owner), np.array(neighbour),
        np.array(tags, dtype=object), zones, len(cells),
    )  # fmt: skip


@dataclass
class built_mesh:
    points: np.ndarray
    faces: list
    owner: np.ndarray
    neighbour: np.ndarray
    tags: np.ndarray
    zones: np.ndarray
    cell_count: int

    def _boundary(self) -> dict[str, np.ndarray]:
        outer = np.flatnonzero(self.neighbour < 0)
        return {
            name: outer[self.tags[outer] == name] for name in sorted(set(self.tags[outer].tolist()))
        }

    def patch_names(self) -> list[str]:
        return list(self._boundary())

    def patch_size(self, name: str) -> int:
        return len(self._boundary().get(name, ()))

    def patch_points(self, name: str) -> np.ndarray:
        return self.points[sorted({p for f in self._boundary()[name] for p in self.faces[f]})]

    def _face_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        centres = np.empty((len(self.faces), 3))
        areas = np.empty((len(self.faces), 3))
        sizes = np.array([len(f) for f in self.faces])
        for size in np.unique(sizes):
            index = np.flatnonzero(sizes == size)
            corner = self.points[np.array([self.faces[f] for f in index])]
            middle = corner.mean(axis=1, keepdims=True)
            normal = 0.5 * np.cross(corner - middle, np.roll(corner, -1, axis=1) - middle)
            weight = np.linalg.norm(normal, axis=2, keepdims=True)
            triangle = (corner + np.roll(corner, -1, axis=1) + middle) / 3
            total = np.maximum(weight.sum(axis=1), 1e-300)
            centres[index] = (triangle * weight).sum(axis=1) / total
            areas[index] = normal.sum(axis=1)
        return centres, areas

    def cell_volumes(self) -> np.ndarray:
        centres, areas = self._face_geometry()
        flux = np.einsum("ij,ij->i", centres, areas) / 3
        volume = np.zeros(self.cell_count)
        np.add.at(volume, self.owner, flux)
        inner = self.neighbour >= 0
        np.add.at(volume, self.neighbour[inner], -flux[inner])
        return volume

    def closure_error(self) -> np.ndarray:
        """Sum of the outward area vectors of each cell, zero for a closed cell."""
        areas = self._face_geometry()[1]
        total = np.zeros((self.cell_count, 3))
        np.add.at(total, self.owner, areas)
        inner = self.neighbour >= 0
        np.add.at(total, self.neighbour[inner], -areas[inner])
        return total

    def require_positive_volumes(self) -> None:
        bad = int((self.cell_volumes() <= 0).sum())
        if bad:
            raise ValueError(f"{bad} inverted cells in the structured mesh")

    def write(self, case: Path, patches: Mapping[str, Mapping[str, object]]) -> None:
        """Write ``constant/polyMesh``. ``patches`` gives the boundary entries of a patch
        (``type`` and any coupling keys); patches not listed are walls. An entry with
        ``copy_of`` repeats the faces of another patch, as blockage patches need.
        """
        boundary = self._boundary()
        copies = {n: str(e["copy_of"]) for n, e in patches.items() if "copy_of" in e}
        missing = set(patches) - set(boundary) - set(copies)
        if missing:
            raise ValueError(f"patches without faces: {sorted(missing)}")
        order = list(boundary)
        for name, source in copies.items():
            order.insert(order.index(source) + 1, name)
            boundary[name] = boundary[source]
        inner = np.flatnonzero(self.neighbour >= 0)
        low = np.minimum(self.owner[inner], self.neighbour[inner])
        high = np.maximum(self.owner[inner], self.neighbour[inner])
        inner = inner[np.lexsort((high, low))]
        sequence = np.concatenate([inner, *(boundary[n] for n in order)])
        directory = Path(case) / "constant/polyMesh"
        directory.mkdir(parents=True, exist_ok=True)
        _list(directory / "points", "vectorField",
              (f"({x:.12g} {y:.12g} {z:.12g})" for x, y, z in self.points), len(self.points))
        _list(directory / "faces", "faceList",
              (f"{len(self.faces[f])}({' '.join(map(str, self.faces[f]))})" for f in sequence),
              len(sequence))  # fmt: skip
        _list(directory / "owner", "labelList", map(str, self.owner[sequence]), len(sequence))
        _list(directory / "neighbour", "labelList", map(str, self.neighbour[inner]), len(inner))
        entries, start = [], len(inner)
        for name in order:
            settings = {k: v for k, v in patches.get(name, {}).items() if k != "copy_of"}
            settings.setdefault("type", "wall")
            lines = "".join(f"        {k} {v};\n" for k, v in settings.items())
            count = len(boundary[name])
            entries.append(
                f"    {name}\n    {{\n{lines}        nFaces {count};\n"
                f"        startFace {start};\n    }}"
            )
            start += count
        _list(directory / "boundary", "polyBoundaryMesh", entries, len(order))
        self._write_zones(directory)

    def _write_zones(self, directory: Path) -> None:
        names = sorted({z for z in self.zones.tolist() if z is not None})
        if not names:
            return
        entries = []
        for name in names:
            cells = np.flatnonzero(self.zones == name)
            body = "\n".join(map(str, cells))
            entries.append(
                f"{name}\n{{\n    type cellZone;\n    cellLabels List<label>\n"
                f"{len(cells)}\n(\n{body}\n);\n}}"
            )
        _list(directory / "cellZones", "regIOobject", entries, len(names))


def _list(path: Path, kind: str, items, count: int) -> None:
    header = (
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class {kind};\n    object {path.name};\n}}\n\n"
    )
    path.write_text(header + f"{count}\n(\n" + "\n".join(items) + "\n)\n")
