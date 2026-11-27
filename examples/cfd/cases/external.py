"""External aerodynamics of bodies meshed with ``snappyHexMesh``.

All geometry is generated here, except the motorbike surface, which is read from the
OpenFOAM installation at run time.
"""

import gzip
import os
import re
from itertools import pairwise
from math import cos, pi, radians, sin, tan
from pathlib import Path

import numpy as np
from foam import measure as post
from foam.boundaries import rotating_wall, wind_profile_inlet
from foam.case import at_least, at_most, cfd_case, variable
from foam.shapes import box, loft, prism, revolve, sphere, volume, wing
from foam.snappy import surface

from .external_flow import DENSITY, wind_tunnel


def _ahmed_outline(slant_deg: float, slant: float, diffuser_deg: float, nose: float):
    """Convex side view of the Ahmed body, 1.044 m long on a 0.05 m ground clearance."""
    length, low, high, diffuser = 1.044, 0.05, 0.338, 0.2
    angle = np.linspace(0, np.pi / 2, 9)
    lower_nose = [(nose * (1 - np.cos(a)), low + nose * (1 - np.sin(a))) for a in angle]
    upper_nose = [(nose * (1 - np.sin(a)), high - nose * (1 - np.cos(a))) for a in angle]
    tail = [
        (length - diffuser, low),
        (length, low + diffuser * tan(radians(diffuser_deg))),
        (length, high - slant * np.sin(radians(slant_deg))),
        (length - slant * np.cos(radians(slant_deg)), high),
    ]
    return [*lower_nose, *tail, *upper_nose]


def ahmed_body(case, design, level):
    outline = _ahmed_outline(
        design["slant_angle_deg"],
        design["slant_length_m"],
        design["diffuser_angle_deg"],
        design["nose_radius_m"],
    )
    half_width = 0.1945
    body = prism(outline, -0.04, half_width)
    return wind_tunnel(
        case,
        level,
        {"body": surface(body, (3, 4))},
        (-2.0, 0.0, 0.0),
        (5.0, 1.5, 1.5),
        0.16,
        speed=40.0,
        length=1.044,
        area=0.389 * 0.288,
        half=True,
        refine=[((-0.4, 0.0, 0.0), (2.2, 0.5, 0.6), 2)],
        extra={"volume_m3": abs(volume(prism(outline, 0.0, half_width))) * 2},
    )


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


def _train_body(nose: float, exponent: float, length: float = 5.0) -> list:
    """Cross sections of a train car with equal rounded nose and tail."""
    angle = np.linspace(0, 2 * pi, 40, endpoint=False)
    sections = []
    for x in np.concatenate(
        [np.linspace(0, nose, 14), np.linspace(nose, length - nose, 4)[1:-1],
         np.linspace(length - nose, length, 14)]
    ):  # fmt: skip
        reach = min(x, length - x, nose) / nose
        scale = max(reach**exponent, 0.04)
        # a rounded rectangle 0.5 m wide and 0.6 m tall
        shape_y = np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** 0.4
        shape_z = np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** 0.4
        sections.append(
            np.column_stack(
                [np.full(40, x), 0.25 * scale * shape_y, 0.1 + 0.3 * scale * (shape_z + 1)]
            )
        )
    return sections


def _turn(triangles: np.ndarray, axis: int, angle: float, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Rotate triangles about a coordinate axis through ``centre``."""
    c, s = cos(angle), sin(angle)
    a, b = [(1, 2), (2, 0), (0, 1)][axis]
    shifted = triangles - np.asarray(centre)
    result = shifted.copy()
    result[..., a] = c * shifted[..., a] - s * shifted[..., b]
    result[..., b] = s * shifted[..., a] + c * shifted[..., b]
    return result + np.asarray(centre)


def train_aerodynamics(case, design, level):
    body = loft(_train_body(design["nose_length_m"], design["nose_exponent"]))
    yawed = _turn(body, 2, radians(design["yaw_deg"]), (2.5, 0.0, 0.0))
    return wind_tunnel(
        case, level, {"train": surface(yawed, (3, 3))},
        (-3.0, -2.0, 0.0), (12.0, 2.5, 2.4), 0.22, speed=45.0, length=5.0, area=0.5 * 0.6,
        refine=[((-0.5, -0.8, 0.0), (7.0, 1.0, 1.0), 1)],
        extra={"volume_m3": volume(body)},
    )  # fmt: skip


def spitfire(case, design, level):
    """A simplified single-engine fighter airframe with an elliptical wing, half model."""
    angle = radians(design["angle_of_attack_deg"])
    washout = design["wing_twist_deg"]
    fuselage = revolve(
        [0.0, 0.5, 1.5, 3.0, 5.0, 7.0, 9.1], [0.05, 0.42, 0.6, 0.62, 0.45, 0.25, 0.04]
    )
    stations = []
    for y in np.linspace(-0.2, 5.6, 14):
        fraction = max(y, 0.0) / 5.6
        chord = 2.4 * np.sqrt(max(1 - fraction**2, 0.0)) + 0.25
        stations.append((y, 3.6 - 0.45 * chord, -0.25 + 0.08 * fraction * 5.6,
                         chord, washout * fraction))  # fmt: skip
    main = wing(stations, thickness=0.13, camber=0.02)
    tail = wing(
        [(-0.1, 7.8, 0.1, 1.2, 0.0), (1.2, 8.0, 0.1, 0.9, 0.0), (1.7, 8.3, 0.1, 0.4, 0.0)],
        thickness=0.1, camber=0.0,
    )  # fmt: skip
    fin = prism([(7.7, 0.1), (9.0, 0.1), (9.0, 1.4), (8.5, 1.4)], -0.1, 0.05)
    parts = {"fuselage": fuselage, "wing": main, "tailplane": tail, "fin": fin}
    surfaces = {
        name: surface(_turn(t, 1, angle, (3.6, 0.0, 0.0)), (4, 4) if name != "fuselage" else (3, 4))
        for name, t in parts.items()
    }
    return wind_tunnel(
        case, level, surfaces, (-12.0, 0.0, -10.0), (30.0, 14.0, 10.0), 1.0,
        speed=60.0, length=9.1, area=22.5, half=True, ground=False,
        refine=[((-1.0, 0.0, -1.5), (13.0, 6.5, 2.0), 2)], inside=(-7.013, 9.027, 5.031),
    )  # fmt: skip


def building_wind_load(case, design, level):
    """A 1:200 tower, tapered, twisted and with chamfered corners, in a wind profile."""
    height, half = 1.0, 0.13
    taper, twist = design["taper_ratio"], radians(design["twist_deg"])
    cut = design["chamfer_fraction"] * 2 * half
    corner = np.array(
        [
            (half, -half + cut), (half, half - cut), (half - cut, half), (-half + cut, half),
            (-half, half - cut), (-half, -half + cut), (-half + cut, -half),
            (half - cut, -half),
        ]
    )  # fmt: skip
    sections = []
    for z in np.linspace(0.0, height, 25):
        fraction = z / height
        scale = 1 - (1 - taper) * fraction
        c, s = cos(twist * fraction), sin(twist * fraction)
        xy = scale * corner @ np.array([[c, s], [-s, c]])
        sections.append(np.column_stack([xy, np.full(len(corner), z)]))
    tower = loft(sections)
    return wind_tunnel(
        case, level, {"tower": surface(tower, (3, 3))},
        (-2.5, -2.0, 0.0), (6.0, 2.0, 2.6), 0.18, speed=10.0, length=height,
        area=2 * half * height, refine=[((-0.5, -0.6, 0.0), (1.8, 0.6, 1.3), 2)],
        overrides={"inlet": wind_profile_inlet(10.0, height, 0.005)},
        libraries=("atmosphericModels",), extra={"volume_m3": volume(tower)},
    )  # fmt: skip


BALL_RADIUS = 0.02134


def _dimples(depth: float, count: int = 90):
    """Smooth spherical caps spread on a Fibonacci lattice."""
    n = np.arange(count) + 0.5
    z = 1 - 2 * n / count
    azimuth = pi * (1 + 5**0.5) * n
    ring = np.sqrt(1 - z**2)
    centres = np.column_stack([ring * np.cos(azimuth), ring * np.sin(azimuth), z])
    half_angle = 0.14

    def relief(direction: np.ndarray) -> np.ndarray:
        nearest = np.arccos(np.clip(direction @ centres.T, -1, 1)).min(axis=1)
        return depth * np.clip(1 - (nearest / half_angle) ** 2, 0, None)

    return relief


def dimpled_ball(case, design, level):
    spin = design["spin_rad_s"]
    ball = sphere(BALL_RADIUS, segments=96, relief=_dimples(design["dimple_depth_m"]))
    return wind_tunnel(
        case, level, {"ball": surface(ball, (4, 4))},
        (-0.2, -0.18, -0.18), (0.55, 0.18, 0.18), 0.024, speed=40.0, length=2 * BALL_RADIUS,
        area=pi * BALL_RADIUS**2, ground=False,
        refine=[((-0.04, -0.04, -0.04), (0.14, 0.04, 0.04), 3)],
        overrides={"ball": rotating_wall(spin, axis=(0, 1, 0))},
        time_step=2e-5, compressible=True, inside=(-0.1513, 0.1117, 0.1231),
    )  # fmt: skip


def beam_ball(case, design, level):
    """A sphere on a cantilever rod in a stream; deflection from beam theory."""
    radius, length = design["ball_radius_m"], design["beam_length_m"]
    rod_radius, modulus = 0.004, 70.0e9
    ball = sphere(radius, centre=(0.0, 0.0, length), segments=48)
    angle = np.linspace(0, 2 * pi, 20, endpoint=False)
    rod = loft(
        [
            np.column_stack(
                [rod_radius * np.cos(angle), rod_radius * np.sin(angle), np.full(20, z)]
            )
            for z in (-0.01, length)
        ]
    )
    speed = 30.0
    functions = post.forces("ball", ["ball"], DENSITY) | post.forces("rod", ["rod"], DENSITY)

    def deflection(directory, metrics):
        tip = float(post.last_force(directory, "ball")[0])
        spread = float(post.last_force(directory, "rod")[0])
        stiffness = modulus * pi * rod_radius**4 / 4
        bend = tip * length**3 / (3 * stiffness) + spread * length**3 / (8 * stiffness)
        return {"deflection_m": bend, "drag_n": tip + spread}

    return wind_tunnel(
        case, level, {"ball": surface(ball, (3, 4)), "rod": surface(rod, (4, 4))},
        (-0.3, -0.25, 0.0), (0.8, 0.25, 0.5), 0.032, speed=speed, length=2 * radius,
        area=pi * radius**2, refine=[((-0.06, -0.06, 0.0), (0.2, 0.06, length + 0.06), 2)],
        functions=functions, derive=deflection, inside=(-0.2013, 0.1517, 0.4131),
    )  # fmt: skip


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


def formula_one(case, design, level):
    """A simplified open-wheel racing car, half model."""
    ride = design["ride_height_m"]
    front = -design["front_wing_deg"]
    rear = -design["rear_wing_deg"]
    nose = prism(
        [(0.5, ride + 0.2), (1.8, ride + 0.05), (4.4, ride + 0.05), (4.4, ride + 0.5),
         (3.0, ride + 0.75), (1.8, ride + 0.5)], -0.05, 0.35,
    )  # fmt: skip
    floor = box((1.6, -0.05, ride), (4.3, 0.75, ride + 0.07))
    front_wing = wing(
        [(-0.05, 0.0, ride + 0.07, 0.55, front), (0.95, 0.0, ride + 0.07, 0.55, front)],
        thickness=0.12, camber=-0.07,
    )  # fmt: skip
    rear_wing = wing(
        [(-0.05, 4.55, 0.85, 0.45, rear), (0.5, 4.55, 0.85, 0.45, rear)],
        thickness=0.12, camber=-0.09,
    )  # fmt: skip
    pylon = box((4.55, -0.05, ride + 0.4), (4.75, 0.06, 0.87))
    wheels = np.concatenate([_wheel(1.0, 0.65, 0.95, 0.33), _wheel(4.0, 0.7, 1.05, 0.34)])
    axles = np.concatenate(
        [box((0.95, 0.0, 0.28), (1.05, 0.7, 0.36)), box((3.95, 0.0, 0.28), (4.05, 0.75, 0.36))]
    )
    parts = {
        "body": nose, "floor": floor, "front_wing": front_wing, "rear_wing": rear_wing,
        "pylon": pylon, "wheels": wheels, "axles": axles,
    }  # fmt: skip
    return wind_tunnel(
        case, level,
        {n: surface(t, (4, 4) if "wing" in n else (3, 4)) for n, t in parts.items()},
        (-6.0, 0.0, 0.0), (18.0, 5.0, 4.0), 0.4, speed=60.0, length=5.0, area=1.4, half=True,
        refine=[((-0.8, 0.0, 0.0), (7.5, 1.6, 1.5), 2)],
    )  # fmt: skip


def drivaer(case, design, level):
    """An own simplified notchback sedan inspired by the DrivAer shape family, half model."""
    slant = radians(design["rear_slant_deg"])
    diffuser = radians(design["diffuser_angle_deg"])
    deck = design["trunk_height_m"]
    floor, tail = 0.16, 4.6
    lower = [
        (0.0, 0.42), (0.18, 0.22), (0.7, floor), (3.8, floor),
        (tail, floor + 0.8 * tan(diffuser)), (tail, deck), (3.9, deck + 0.03),
        (1.15, 0.92), (0.15, 0.68),
    ]  # fmt: skip
    roof, roof_start, roof_end = 1.42, 2.05, 2.85
    rear_foot = min(roof_end + (roof - deck + 0.05) / tan(slant), tail - 0.1)
    cabin = [
        (1.05, 0.85), (rear_foot, 0.85), (rear_foot, deck - 0.02), (roof_end, roof),
        (roof_start, roof),
    ]  # fmt: skip
    wheels = np.concatenate([_wheel(0.95, 0.62, 0.92, 0.32), _wheel(3.75, 0.62, 0.92, 0.32)])
    axles = np.concatenate(
        [box((0.9, 0.0, 0.27), (1.0, 0.68, 0.34)), box((3.7, 0.0, 0.27), (3.8, 0.68, 0.34))]
    )
    shell, top = prism(lower, -0.05, 0.82), prism(cabin, -0.05, 0.7)
    parts = {"body": shell, "cabin": top, "wheels": wheels, "axles": axles}
    size = abs(volume(prism(lower, 0.0, 0.82))) * 2 + abs(volume(prism(cabin, 0.0, 0.7))) * 2
    return wind_tunnel(
        case, level, {n: surface(t, (3, 4)) for n, t in parts.items()},
        (-7.0, 0.0, 0.0), (18.0, 5.0, 4.5), 0.4, speed=40.0, length=4.6, area=2.16, half=True,
        refine=[((-0.8, 0.0, 0.0), (7.5, 1.5, 1.9), 2)], extra={"volume_m3": size},
    )  # fmt: skip


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    return cfd_case(
        name, "external", title, "snappy", solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip


_DRAG = ("drag_coefficient",)

CASES = [
    _case(
        "ahmed_body", "Ahmed reference car body, half model", ahmed_body,
        (
            variable("slant_angle_deg", 12.0, 35.0, 25.0),
            variable("slant_length_m", 0.15, 0.25, 0.222),
            variable("diffuser_angle_deg", 0.0, 8.0, 2.0),
            variable("nose_radius_m", 0.06, 0.12, 0.1),
        ),
        "drag_coefficient", (at_least("volume_m3", 0.03),), positive=_DRAG,
        simplification="no support stilts, rounding only in side view",
    ),
    _case(
        "motorbike_benchmark", "Solver run time on the motorbike geometry", motorbike_benchmark,
        (variable("surface_level", 3.0, 5.0, 4.0), variable("background_cell_m", 0.3, 0.6, 0.4)),
        "execution_time_s", (at_most("drag_coefficient", 0.25),), positive=_DRAG,
        simplification="speed benchmark; processes are set with --cores",
    ),
    _case(
        "train_aerodynamics", "Train car nose shape and yaw", train_aerodynamics,
        (
            variable("nose_length_m", 0.5, 1.4, 0.9),
            variable("nose_exponent", 0.35, 0.8, 0.55),
            variable("yaw_deg", 0.0, 10.0, 3.0),
        ),
        "drag_coefficient", (at_least("volume_m3", 0.05),), positive=_DRAG,
        simplification="single car, stationary ground",
    ),
    _case(
        "spitfire", "Fighter airframe lift and drag", spitfire,
        (
            variable("angle_of_attack_deg", 1.0, 10.0, 4.0),
            variable("wing_twist_deg", -4.0, 0.0, -2.0),
        ),
        "lift_to_drag", (at_least("lift_coefficient", 0.2),), maximize=True, positive=_DRAG,
        simplification="simplified airframe, no propeller",
    ),
    _case(
        "building_wind_load", "Tall building in an atmospheric wind profile", building_wind_load,
        (
            variable("taper_ratio", 0.6, 1.0, 0.85),
            variable("twist_deg", 0.0, 90.0, 30.0),
            variable("chamfer_fraction", 0.02, 0.2, 0.08),
        ),
        "base_moment_coefficient", (at_least("volume_m3", 0.05),),
        positive=("base_moment_coefficient", "drag_coefficient"),
        simplification="generic tower, steady wind",
    ),
    _case(
        "dimpled_ball", "Spinning dimpled ball", dimpled_ball,
        (variable("dimple_depth_m", 0.0002, 0.001, 0.0005),
         variable("spin_rad_s", 200.0, 700.0, 628.0)),
        "drag_coefficient", (at_least("lift_coefficient", 0.25),), solver="rhoPimpleFoam",
        positive=_DRAG,
        simplification="short transient, spinning surface without moving dimples, no acoustics",
    ),
    _case(
        "beam_ball", "Ball on a cantilever rod", beam_ball,
        (variable("ball_radius_m", 0.015, 0.035, 0.025),
         variable("beam_length_m", 0.12, 0.3, 0.2)),
        "drag_n", (at_most("deflection_m", 0.2),), positive=("drag_n", "deflection_m"),
        simplification="one-way estimate from beam theory, no fluid-structure coupling",
    ),
    _case(
        "formula_one", "Simplified open-wheel racing car", formula_one,
        (
            variable("front_wing_deg", 2.0, 20.0, 10.0),
            variable("rear_wing_deg", 5.0, 30.0, 16.0),
            variable("ride_height_m", 0.05, 0.12, 0.08),
        ),
        "downforce_coefficient", (at_most("drag_coefficient", 0.15),), maximize=True,
        positive=_DRAG, simplification="simplified body, static wheels and ground",
    ),
    _case(
        "drivaer", "Simplified notchback sedan", drivaer,
        (
            variable("rear_slant_deg", 18.0, 35.0, 26.0),
            variable("diffuser_angle_deg", 0.0, 8.0, 3.0),
            variable("trunk_height_m", 0.9, 1.15, 1.02),
        ),
        "drag_coefficient", (at_least("volume_m3", 0.03),), positive=_DRAG,
        simplification="own geometry, not the published DrivAer model; static wheels and ground",
    ),
]
