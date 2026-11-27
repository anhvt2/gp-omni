"""Structured meshes of bladed machines turning about the z axis.

A machine is a meridional channel between a hub and a shroud curve, given as ``(z, r)``
points, with one or more blade rows. One blade pitch is meshed as an H-grid between the
camber surfaces of neighbouring blades; the pitchwise faces are blade walls inside a row's
chord and periodic elsewhere. ``full_wheel`` repeats the passage around the circle, which a
spiral volute or a sliding mesh needs. All rows have the same blade count.

Blade angles are measured from the meridional direction, positive toward increasing theta.
"""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .case import mesh_level
from .hex_mesh import hex_mesh

ANGLE_LIMIT = 68.0  # degrees; larger angles shear the grid too much


@dataclass(frozen=True)
class blade_row:
    name: str
    start: float  # leading edge, fraction of the meridional length
    end: float  # trailing edge
    # degrees at (hub, tip), or a function of the blade height fraction
    inlet_angle: tuple[float, float] | Callable[[float], float]
    outlet_angle: tuple[float, float] | Callable[[float], float]
    thickness: float = 0.1  # largest blockage as a fraction of the pitch
    omega: float = 0.0  # rad/s; non-zero rows turn
    span: float = 1.0  # blade height as a fraction of the channel height


@dataclass(frozen=True)
class volute:
    """A rectangular spiral casing joined to the outlet (``at="end"``) or inlet radius."""

    height: float  # smallest radial height
    growth: float  # extra height reached after almost one turn
    at: str = "end"
    offset: float = 0.0  # axial shift as a fraction of the width, for partial overlap


@dataclass(frozen=True)
class machine:
    hub: tuple
    shroud: tuple
    blades: int
    rows: tuple[blade_row, ...]
    outer: str = "shroud"  # "farfield" for open rotors
    full_wheel: bool = False
    spiral: volute | None = None
    sliding: bool = False  # separate rotating and stationary regions

    @property
    def sectors(self) -> int:
        """Number of identical passages the meshed domain represents once."""
        return 1 if self.full_wheel else self.blades


def axial_channel(length: float, hub_radius: float, tip_radius: float, points: int = 41):
    z = np.linspace(0.0, length, points)
    hub = tuple(zip(z, np.full(points, hub_radius)))
    shroud = tuple(zip(z, np.full(points, tip_radius)))
    return hub, shroud


def radial_channel(
    eye_hub: float,
    eye_tip: float,
    outer: float,
    width: float,
    depth: float,
    inlet: float,
    outlet: float = 0.0,
    points: int = 61,
):
    """Axial entry bending to a radial exit of ``width`` at radius ``outer + outlet``."""
    t = np.linspace(0.0, np.pi / 2, points)

    def side(radius: float, reach: float):
        z = np.concatenate([[-inlet], reach * np.sin(t)])
        r = np.concatenate([[radius], outer - (outer - radius) * np.cos(t)])
        if outlet > 0:
            z, r = np.append(z, reach), np.append(r, outer + outlet)
        return tuple(zip(z, r))

    return side(eye_hub, depth), side(eye_tip, depth - width)


def _along(curve, fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Points of a polyline at fractions of its arc length."""
    points = np.asarray(curve, dtype=float)
    length = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    return tuple(np.interp(fraction * length[-1], length, points[:, d]) for d in range(2))


def _segments(breaks: list[float], counts: list[int]) -> tuple[np.ndarray, list[int]]:
    """Nodes through all breakpoints, and the node number of each breakpoint."""
    nodes, marks = [np.array([breaks[0]])], [0]
    for low, high, count in zip(breaks[:-1], breaks[1:], counts):
        nodes.append(np.linspace(low, high, count + 1)[1:])
        marks.append(marks[-1] + count)
    return np.concatenate(nodes), marks


def _streamwise(design: machine, level: mesh_level):
    rows = design.rows
    for first, second in pairwise(rows):
        if second.start <= first.end:
            raise ValueError("blade rows must not overlap")
    if rows[0].start <= 0 or rows[-1].end >= 1 or any(r.end <= r.start for r in rows):
        raise ValueError("blade rows must lie inside the channel")
    breaks = [0.0, *(v for row in rows for v in (row.start, row.end)), 1.0]
    counts = []
    for number, (low, high) in enumerate(pairwise(breaks)):
        in_blade = number % 2 == 1
        base = 28 if in_blade else max(5, round(40 * (high - low)))
        counts.append(level.cells(base, 4))
    # nodes midway between rows separate their rotating zones
    return _segments(breaks, counts), breaks


def _spanwise(design: machine, level: mesh_level):
    tips = sorted({row.span for row in design.rows if row.span < 1})
    breaks = [0.0, *tips, 1.0]
    counts = [level.cells(12 if number == 0 else 8, 3) for number in range(len(breaks) - 1)]
    return _segments(breaks, counts)[0]


def _angle_at(given, height: float) -> float:
    if callable(given):
        return given(height)
    hub, tip = given
    return hub + (tip - hub) * height


def _angles(design: machine, u: np.ndarray, s: np.ndarray, breaks: list[float]) -> np.ndarray:
    """Camber angle in radians at every meridional node ``(i, j)``."""
    result = np.empty((len(u), len(s)))
    for j, span in enumerate(s):
        knots_u, knots_angle = [0.0], [0.0]
        for row in design.rows:
            local = min(span / row.span, 1.0)
            inlet, outlet = (
                np.clip(_angle_at(given, local), -ANGLE_LIMIT, ANGLE_LIMIT)
                for given in (row.inlet_angle, row.outlet_angle)
            )
            fade = 1.0 if span <= row.span else max(0.0, 1 - (span - row.span) / (1 - row.span))
            knots_u += [row.start, row.end]
            knots_angle += [inlet * fade, outlet * fade]
        first, last = design.rows[0], design.rows[-1]
        chord_in, chord_out = first.end - first.start, last.end - last.start
        # return to the meridional direction within half a chord of the outer edges
        knots_u.insert(1, max(first.start - 0.5 * chord_in, 0.5 * first.start))
        knots_angle.insert(1, 0.0)
        knots_u += [min(last.end + 0.5 * chord_out, 0.5 * (1 + last.end)), 1.0]
        if design.spiral and design.spiral.at == "end":
            knots_angle += [knots_angle[-1], knots_angle[-1]]
        else:
            knots_angle += [0.0, 0.0]
        result[:, j] = np.interp(u, knots_u, knots_angle)
    return np.radians(result)


@dataclass
class passage_grid:
    z: np.ndarray  # (ni, nj)
    r: np.ndarray
    camber: np.ndarray  # theta of the blade mean surface
    gap: np.ndarray  # blade thickness as an angle
    pitch: float
    v: np.ndarray  # pitchwise fractions


def passage_points(design: machine, level: mesh_level):
    (u, marks), breaks = _streamwise(design, level)
    s = _spanwise(design, level)
    hub_z, hub_r = _along(design.hub, u)
    tip_z, tip_r = _along(design.shroud, u)
    z = hub_z[:, None] * (1 - s) + tip_z[:, None] * s
    r = hub_r[:, None] * (1 - s) + tip_r[:, None] * s
    angle = _angles(design, u, s, breaks)
    step = np.hypot(np.diff(z, axis=0), np.diff(r, axis=0))
    turn = 0.5 * (np.tan(angle[1:]) + np.tan(angle[:-1])) * step / (0.5 * (r[1:] + r[:-1]))
    camber = np.vstack([np.zeros(len(s)), np.cumsum(turn, axis=0)])
    if design.spiral:  # two-dimensional blades, so the casing joint is a surface of revolution
        camber = np.repeat(camber[:, [len(s) // 2]], len(s), axis=1)
    middle = (marks[1] + marks[2]) // 2
    camber -= camber[middle]  # stack the blade sections at mid chord
    pitch = 2 * np.pi / design.blades
    gap = np.zeros_like(camber)
    for number, row in enumerate(design.rows):
        low, high = marks[2 * number + 1], marks[2 * number + 2]
        x = np.linspace(0, 1, high - low + 1)
        profile = row.thickness * pitch * 4 * x * (1 - x)
        covered = s < row.span - 1e-9 if row.span < 1 else np.ones(len(s), bool)
        gap[low : high + 1] = profile[:, None] * covered
    count = level.cells(18, 6)
    even = np.linspace(0, 1, count + 1)
    v = 0.5 * even + 0.5 * (0.5 - 0.5 * np.cos(np.pi * even))  # refine toward the blades
    return passage_grid(z, r, camber, gap, pitch, v), marks, s, u


def _cartesian(r, theta, z) -> np.ndarray:
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)


def _zones(design: machine, marks, s, cells) -> tuple[np.ndarray, np.ndarray]:
    """Zone name of every passage cell and the zone-edge node numbers of each rotor."""
    zones = np.full(cells, None, dtype=object)
    edges = np.zeros((len(design.rows), 2), dtype=int)
    for number, row in enumerate(design.rows):
        low, high = marks[2 * number + 1], marks[2 * number + 2]
        before = marks[2 * number]
        after = marks[2 * number + 3]
        start = low - (low - before) // 2
        stop = high + (after - high) // 2
        edges[number] = start, stop
        if row.omega == 0:
            continue
        radial = np.ones(cells[1], bool)
        if row.span < 1:
            limit = min(1.0, row.span + 0.35 * (1 - row.span))
            radial = 0.5 * (s[1:] + s[:-1]) < limit
        zones[start:stop, radial, :] = f"zone_{row.name}"
    return zones, edges


def build_mesh(design: machine, level: mesh_level) -> hex_mesh:
    if design.spiral and not design.full_wheel:
        raise ValueError("a volute needs the full wheel")
    grid, marks, s, _ = passage_points(design, level)
    ni, nj = (n - 1 for n in grid.z.shape)
    nk = len(grid.v) - 1
    zones, edges = _zones(design, marks, s, (ni, nj, nk))
    mid_span = 0.5 * (s[1:] + s[:-1])

    sides = {}
    for side, periodic in (("k0", "periodic_a"), ("k1", "periodic_b")):
        names = np.full((ni, nj), "periodic" if design.full_wheel else periodic, dtype=object)
        for number, row in enumerate(design.rows):
            low, high = marks[2 * number + 1], marks[2 * number + 2]
            names[low:high, mid_span < row.span] = f"blade_{row.name}"
        sides[side] = names
    hub = np.full((ni, nk), "hub", dtype=object)
    for number, row in enumerate(design.rows):
        if row.omega:
            hub[slice(*edges[number])] = f"hub_{row.name}"

    mesh = hex_mesh()
    wheel = range(design.blades) if design.full_wheel else range(1)
    rotor = next((n for n, row in enumerate(design.rows) if row.omega), 0)
    for m in wheel:
        theta = (
            grid.camber[:, :, None]
            + 0.5 * grid.gap[:, :, None]
            + grid.v[None, None, :] * (grid.pitch - grid.gap[:, :, None])
            + m * grid.pitch
        )
        points = _cartesian(grid.r[:, :, None], theta, grid.z[:, :, None] + 0 * theta)
        ends = {"i0": "inlet", "i1": "outlet", "j0": hub, "j1": design.outer, **sides}
        if design.sliding and design.spiral is None:
            start, stop = edges[rotor]
            pieces = (
                (0, start, 0, {"i0": "inlet", "i1": "front_static"}),
                (start, stop, 1, {"i0": "front_rotor", "i1": "rear_rotor"}),
                (stop, ni, 0, {"i0": "rear_static", "i1": "outlet"}),
            )
            for low, high, region, caps in pieces:
                local = {
                    "j0": hub[low:high], "j1": design.outer,
                    "k0": sides["k0"][low:high], "k1": sides["k1"][low:high], **caps,
                }  # fmt: skip
                name = "rotor" if region else None
                if region:
                    local["j0"] = np.full((high - low, nk), "hub_rotor", object)
                    local["j1"] = "shroud_rotor"
                mesh.add_block(points[low : high + 1], local, zone=name, region=region)
            continue
        zone = zones
        region = 0
        if design.sliding:  # the whole impeller turns inside a stationary volute
            zone, region = "rotor", 1
            ends["i1"] = "rotor_couple"
        mesh.add_block(points, ends, zone=zone, region=region)
        if design.spiral:
            _add_volute(mesh, design, grid, m, level)
    return mesh


def _add_volute(mesh: hex_mesh, design: machine, grid: passage_grid, m: int, level: mesh_level):
    shape = design.spiral
    at_end = shape.at == "end"
    index = -1 if at_end else 0
    camber, z, radius = grid.camber[index], grid.z[index], grid.r[index]
    if np.ptp(camber) > 1e-9 or np.ptp(radius) > 1e-9:
        raise ValueError("a volute needs two-dimensional blades and a constant joint radius")
    theta = camber[0] + (m + grid.v) * grid.pitch
    turn = np.mod(theta - camber[0], 2 * np.pi)
    turn = np.where((grid.v >= 1 - 1e-12) & (m == design.blades - 1), 2 * np.pi, turn)
    ramp = min(2 * grid.pitch, np.pi / 2)
    rise = np.where(
        turn <= 2 * np.pi - ramp, turn / (2 * np.pi - ramp), (2 * np.pi - turn) / ramp
    )
    height = shape.height + shape.growth * rise
    cells = level.cells(8, 3)
    fraction = np.linspace(0, 1, cells + 1)
    if not at_end:
        fraction = fraction[::-1]  # from the outer wall inward to the joint
    r = radius[0] + fraction[:, None, None] * height[None, None, :]
    width = z[-1] - z[0]
    axial = z + shape.offset * width
    shape3 = (cells + 1, len(z), len(theta))
    points = _cartesian(
        np.broadcast_to(r, shape3),
        np.broadcast_to(theta[None, None, :], shape3),
        np.broadcast_to(axial[None, :, None], shape3),
    )
    middle = 0.5 * (turn[1:] + turn[:-1])
    # a discharge slot over half a turn before the casing closes keeps its speed moderate
    window = (middle > np.pi - ramp) & (middle <= 2 * np.pi - ramp)
    port = "outlet" if at_end else "inlet"
    outer = np.where(window, port, "casing").astype(object)
    outer_face = np.broadcast_to(outer[None, :], (len(z) - 1, len(theta) - 1))
    joint = "volute_couple" if design.sliding else "joint"
    sides = {"j0": "casing", "j1": "casing", "k0": "volute_cut", "k1": "volute_cut"}
    sides |= (
        {"i0": joint, "i1": outer_face} if at_end else {"i0": outer_face, "i1": joint}
    )
    mesh.add_block(points, sides)
