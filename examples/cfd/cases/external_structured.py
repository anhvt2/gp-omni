"""External flows on structured meshes: plate, airfoil and bullet."""

from math import cos, pi, radians, sin, tan

import numpy as np
from foam import measure as post
from foam.boundaries import (
    empty,
    freestream,
    open_outlet,
    pressure_outlet,
    slip,
    supersonic_inlet,
    symmetry,
    velocity_inlet,
    wall,
    wedge,
)
from foam.case import at_least, at_most, cfd_case, run_plan, variable
from foam.hex_mesh import hex_mesh, lattice
from foam.physics import flow, write_flow_case
from foam.shapes import naca_outline

from .internal import WEDGES, _finish, _wedge_grid

AIR_DENSITY = 1.2
VISCOSITY = 1.5e-5


def _graded(first: float, length: float, cells: int) -> np.ndarray:
    """Nodes from 0 to ``length`` growing geometrically from a first cell of ``first``."""
    if first * cells >= length:
        return np.linspace(0, length, cells + 1)
    low, high = 1.0 + 1e-9, 2.0
    for _ in range(100):
        ratio = 0.5 * (low + high)
        total = first * (ratio**cells - 1) / (ratio - 1)
        low, high = (ratio, high) if total < length else (low, ratio)
    steps = first * ratio ** np.arange(cells)
    nodes = np.concatenate([[0.0], np.cumsum(steps)])
    return nodes * length / nodes[-1]


def flat_plate(case, design, level):
    speed, first = design["speed_m_s"], design["first_cell_m"]
    length, height, span = 2.0, 1.0, 0.1
    lead = -_graded(0.01, 0.3, level.cells(16))[::-1]
    plate = _graded(0.01, length, level.cells(60))
    x = np.concatenate([lead[:-1], plate])
    y = _graded(first, height, level.cells(50))
    grid = lattice(x, y, np.array([0.0, span]), lambda a, b, c: (a, b, c))
    middle = 0.5 * (x[1:] + x[:-1])
    bottom = np.where(middle > 0, "plate", "upstream").astype(object)
    mesh = hex_mesh()
    mesh.add_block(
        grid,
        {"i0": "inlet", "i1": "outlet", "j0": bottom[:, None], "j1": "top",
         "k0": "front", "k1": "back"},
    )  # fmt: skip
    patches = _finish(
        case,
        mesh,
        {
            "inlet": velocity_inlet((speed, 0, 0)), "outlet": pressure_outlet(),
            "plate": wall(), "upstream": symmetry(), "top": slip(), "front": empty(),
            "back": empty(),
        },
    )  # fmt: skip
    functions = post.flux("inlet", "outlet") | post.forces("plate", ["plate"], AIR_DENSITY)
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam", velocity=speed, length=0.1, iterations=level.iterations,
            second_order=level.resolution > 1, functions=functions,
            initial_velocity=(speed, 0.0, 0.0),
        ),
    )  # fmt: skip
    cells = (len(x) - 1) * (len(y) - 1)
    reynolds = speed * length / VISCOSITY
    correlation = 0.074 / reynolds**0.2

    def measure(directory):
        drag = float(post.last_force(directory, "plate")[0])
        coefficient = drag / (0.5 * AIR_DENSITY * speed**2 * length * span)
        return {
            # the top boundary also passes flow, so inlet and outlet are not compared
            "drag_coefficient": coefficient,
            "drag_error": abs(coefficient - correlation) / correlation,
            "cell_count": float(cells),
        }

    return run_plan(solve=[["simpleFoam"]], measure=measure)


def _airfoil_mesh(level) -> tuple[hex_mesh, float, float]:
    """O-grid around a NACA 0012 section of unit chord in the x-y plane."""
    chord, radius, span = 1.0, 15.0, 0.1
    outline = naca_outline(chord, points=level.cells(60))
    half = level.cells(60)
    upper, lower = outline[:half], outline[half:]
    # start at the trailing edge, run along the upper side to the nose and back below
    loop = np.vstack([upper[::-1], lower[::-1], upper[-1:]])
    count = len(loop) - 1
    angle = 2 * pi * np.arange(count + 1) / count
    far = np.column_stack([0.5 + radius * np.cos(angle), radius * np.sin(angle)])
    t = _graded(2e-3, radius, level.cells(60)) / radius
    plane = loop[:, None, :] * (1 - t[None, :, None]) + far[:, None, :] * t[None, :, None]
    points = np.empty((count + 1, len(t), 2, 3))
    points[..., 0] = plane[:, :, None, 0]
    points[..., 1] = plane[:, :, None, 1]
    points[..., 2] = np.array([0.0, span])[None, None, :]
    mesh = hex_mesh()
    mesh.add_block(
        points,
        {"i0": "cut", "i1": "cut", "j0": "airfoil", "j1": "farfield", "k0": "front",
         "k1": "back"},
    )  # fmt: skip
    return mesh, chord, span


def _airfoil(case, level, angle_deg: float, speed: float, time_step: float = 0.0):
    mesh, chord, span = _airfoil_mesh(level)
    angle = radians(angle_deg)
    velocity = (speed * cos(angle), speed * sin(angle), 0.0)
    patches = _finish(
        case,
        mesh,
        {"airfoil": wall(), "farfield": freestream(velocity), "front": empty(), "back": empty()},
    )
    functions = post.forces("airfoil", ["airfoil"], AIR_DENSITY, (0.25, 0, 0))
    if time_step:
        functions |= post.probes("pressure_probes", ((1.02, 0.03, 0.05), (0.9, 0.12, 0.05)))
    solver = "pimpleFoam" if time_step else "simpleFoam"
    write_flow_case(
        case,
        patches,
        flow(
            solver, velocity=speed, length=0.1 * chord, iterations=level.iterations,
            time_step=time_step, time_steps=level.time_steps,
            second_order=level.resolution > 1, functions=functions,
        ),
    )  # fmt: skip

    def measure(directory):
        force = post.last_force(directory, "airfoil")
        dynamic = 0.5 * AIR_DENSITY * speed**2 * chord * span
        drag = float(force[0] * cos(angle) + force[1] * sin(angle)) / dynamic
        lift = float(force[1] * cos(angle) - force[0] * sin(angle)) / dynamic
        metrics = {
            "lift_coefficient": lift,
            "drag_coefficient": drag,
            "lift_to_drag": lift / drag if abs(drag) > 1e-9 else 0.0,
        }
        if time_step:
            metrics["pressure_fluctuation_pa"] = AIR_DENSITY * post.probe_fluctuation(
                directory, "pressure_probes"
            )
        return metrics

    return run_plan(solve=[[solver]], measure=measure)


def naca_0012(case, design, level):
    return _airfoil(case, level, design["angle_deg"], 51.0)


def naca_0012_acoustic(case, design, level):
    return _airfoil(case, level, design["angle_deg"], design["speed_m_s"], time_step=2e-5)


def bullet_223(case, design, level):
    """A 23 mm bullet at 900 m/s, axisymmetric, with a short closing tail."""
    radius, length = 0.00285, 0.023
    nose, tail = design["nose_length_m"], design["boat_tail_length_m"]
    base = radius - tail * tan(radians(design["boat_tail_angle_deg"]))
    closing = 1.5 * base
    end = length + closing
    x = np.concatenate(
        [
            np.linspace(-0.012, 0.0, level.cells(30))[:-1],
            np.linspace(0.0, end, level.cells(150))[:-1],
            end + _graded(end / 150, 0.05, level.cells(50)),
        ]
    )
    along = np.clip(x / nose, 0, 1)
    # tangent ogive, cylinder, boat tail and closing cone
    rho = (radius**2 + nose**2) / (2 * radius)
    ogive = np.sqrt(np.maximum(rho**2 - (nose * (1 - along)) ** 2, 0)) + radius - rho
    body = np.interp(
        x, [nose, length - tail, length, end], [radius, radius, base, 0.0], right=0.0
    )
    body = np.where(x < nose, ogive, body)
    body = np.where((x <= 0) | (x >= end), 0.0, body)
    outer = np.full(len(x), 0.025)
    mesh = hex_mesh()
    surface = np.where((x[1:] + x[:-1] > 0) & (x[1:] + x[:-1] < 2 * end), "body", "axis")
    mesh.add_block(
        _wedge_grid(x, body, outer, level.cells(50)),
        {
            "i0": "inlet", "i1": "outlet", "j0": surface.astype(object)[:, None], "j1": "far",
            "k0": "wedge_back", "k1": "wedge_front",
        },
    )  # fmt: skip
    speed, pressure, temperature = 900.0, 101325.0, 288.0
    patches = _finish(
        case,
        mesh,
        {
            "inlet": supersonic_inlet((speed, 0, 0), pressure, temperature),
            "outlet": open_outlet(), "far": open_outlet(), "body": wall(), "axis": slip(),
            "wedge_back": wedge(), "wedge_front": wedge(),
        },
    )  # fmt: skip
    functions = post.forces("body", ["body"], None, reference_pressure=pressure)
    functions |= post.flux("inlet", "outlet")
    model = flow(
        "rhoCentralFoam", velocity=speed, length=radius, turbulent=False,
        time_step=4e-8, time_steps=level.time_steps, pressure=pressure,
        temperature=temperature, functions=functions,
    )  # fmt: skip
    write_flow_case(case, patches, model)
    dense = np.linspace(0, end, 2000)
    profile = np.interp(dense, x, body)
    mass = float(np.trapezoid(pi * profile**2, dense)) * 10500.0

    def measure(directory):
        drag = float(post.last_force(directory, "body")[0]) * WEDGES
        dynamic = 0.5 * pressure / (287.0 * temperature) * speed**2 * pi * radius**2
        return {
            "flow_mismatch": post.imbalance(directory, "inlet", "outlet"),
            "drag_coefficient": drag / dynamic,
            "mass_g": 1000.0 * mass,
        }

    return run_plan(solve=[["rhoCentralFoam"]], measure=measure)


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    return cfd_case(
        name, "external", title, "structured", solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip


CASES = [
    _case(
        "flat_plate", "Turbulent flat plate against a drag correlation", flat_plate,
        (variable("speed_m_s", 30.0, 70.0, 50.0), variable("first_cell_m", 2e-4, 3e-3, 1e-3)),
        "drag_error", (at_most("cell_count"),), positive=("drag_coefficient",),
        simplification="operating variables only, a pipeline check",
    ),
    _case(
        "naca_0012", "NACA 0012 airfoil in two dimensions", naca_0012,
        (variable("angle_deg", 1.0, 12.0, 5.0),),
        "lift_to_drag", (at_least("lift_coefficient", 0.2),), maximize=True,
        positive=("drag_coefficient", "lift_coefficient"),
    ),
    _case(
        "naca_0012_acoustic", "Unsteady pressure near the NACA 0012 trailing edge",
        naca_0012_acoustic,
        (variable("angle_deg", 0.0, 10.0, 5.0), variable("speed_m_s", 40.0, 70.0, 55.0)),
        "pressure_fluctuation_pa", (at_least("lift_coefficient", 0.4),), solver="pimpleFoam",
        simplification="two-dimensional short transient, probes only",
    ),
    _case(
        "bullet_223", "Supersonic bullet", bullet_223,
        (
            variable("nose_length_m", 0.008, 0.013, 0.0105),
            variable("boat_tail_length_m", 0.0015, 0.0045, 0.003),
            variable("boat_tail_angle_deg", 4.0, 12.0, 8.0),
        ),
        "drag_coefficient", (at_least("mass_g"),), solver="rhoCentralFoam",
        positive=("drag_coefficient",),
        simplification="axisymmetric short transient, no structural analysis",
    ),
]
