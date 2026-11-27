"""Internal and process flows on structured meshes."""

from math import ceil, pi, radians, tan

import numpy as np
from foam import measure as post
from foam.boundaries import (
    flow_inlet,
    interface,
    moving_wall,
    pressure_outlet,
    rotating_wall,
    slip,
    total_pressure_inlet,
    velocity_inlet,
    wall,
    wedge,
)
from foam.case import at_least, at_most, cfd_case, run_plan, variable
from foam.ducts import sweep, u_bend
from foam.hex_mesh import box_grid, hex_mesh, lattice
from foam.physics import flow, rotation, write_flow_case

WATER = {"density": 1000.0, "viscosity": 1e-6}
HALF_WEDGE = radians(2.5)
WEDGES = 72  # wedges in a full turn


def _finish(case, mesh: hex_mesh, patches: dict) -> dict:
    """Check and write a structured mesh; return the conditions of the patches it has."""
    built = mesh.build()
    built.require_positive_volumes()
    present = {n: patches[n] for n in built.patch_names()}
    built.write(case, {n: b.mesh_entry() for n, b in present.items()})
    return present


def _wedge_grid(x, inner, outer, radial: int) -> np.ndarray:
    """Axisymmetric lattice about x between the radii ``inner(x)`` and ``outer(x)``."""
    t = np.linspace(0, 1, radial + 1)
    t = 0.5 * t + 0.5 * (0.5 - 0.5 * np.cos(pi * t))
    r = np.asarray(inner)[:, None] * (1 - t) + np.asarray(outer)[:, None] * t
    angle = np.array([-HALF_WEDGE, HALF_WEDGE])
    points = np.empty((len(x), len(t), 2, 3))
    points[..., 0] = np.asarray(x)[:, None, None]
    points[..., 1] = r[:, :, None] * np.cos(angle)
    points[..., 2] = r[:, :, None] * np.sin(angle)
    return points


def _bend(case, level, bend_radius: float, radius_at, flow_rate: float, diameter: float):
    """Water through a 180 degree bend; ``radius_at`` takes the path fraction."""
    path, total = u_bend(0.4, bend_radius, 0.8)
    mesh = hex_mesh()
    mesh.add_block(
        sweep(path, radius_at, level.cells(80), level.cells(14, 8)),
        {"i0": "inlet", "i1": "outlet"},
    )
    patches = _finish(
        case,
        mesh,
        {"inlet": flow_inlet(flow_rate), "outlet": pressure_outlet(), "walls": wall()},
    )
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam",
            velocity=flow_rate / (pi * diameter**2 / 4),
            length=diameter,
            iterations=level.iterations,
            second_order=level.resolution > 1,
            functions=post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet"),
            **WATER,
        ),
    )
    s = np.linspace(0, 1, 801)
    volume = float(np.trapezoid(pi * radius_at(s) ** 2, s * total))

    def measure(directory):
        loss = post.last_surface(directory, "total_p_inlet") - post.last_surface(
            directory, "total_p_outlet"
        )
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "pressure_loss_pa": WATER["density"] * loss,
            "volume_m3": volume,
        }

    return run_plan(solve=[["simpleFoam"]], measure=measure)


def pipe(case, design, level):
    diameter = design["diameter_m"]
    return _bend(
        case,
        level,
        design["bend_radius_m"],
        lambda s: np.full(len(s), diameter / 2),
        0.01,
        diameter,
    )


STATIONS = tuple(f"diameter_{n}_m" for n in range(1, 10))


def parametric_pipe(case, design, level):
    radius = 0.25
    _, total = u_bend(0.4, radius, 0.8)
    # nine stations spread evenly through the bend, between fixed 100 mm ends
    along = (0.4 + np.linspace(0.05, 0.95, 9) * pi * radius) / total
    knots = np.concatenate([[0.0, 0.4 / total], along, [(0.4 + pi * radius) / total, 1.0]])
    values = np.array([0.1, 0.1, *(design[n] for n in STATIONS), 0.1, 0.1]) / 2
    return _bend(case, level, radius, lambda s: np.interp(s, knots, values), 0.01, 0.1)


def hydraulic_valve(case, design, level):
    pipe_radius, stem, outlet_radius = 0.01, 0.0085, 0.011
    seat, lift = design["seat_radius_m"], design["lift_m"]
    slope = tan(radians(design["cone_half_angle_deg"]))
    seat_x = 0.03
    tip = seat_x + lift - seat / slope  # the cone leaves a gap of lift*slope at the seat
    shoulder = tip + stem / slope
    x = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, max(tip - 0.002, 0.005), level.cells(30)),
                np.linspace(max(tip - 0.002, 0.005), shoulder + 0.006, level.cells(90)),
                np.linspace(shoulder + 0.006, 0.11, level.cells(50)),
            ]
        )
    )
    outer = np.interp(
        x,
        [0.0, seat_x - 0.008, seat_x - 0.004, seat_x, seat_x + 0.002, 0.11],
        [pipe_radius, pipe_radius, seat, seat, outlet_radius, outlet_radius],
    )
    inner = np.clip((x - tip) * slope, 0.0, stem)
    inner = np.where(x > 0.07, np.clip(stem * (0.085 - x) / 0.015, 0.0, stem), inner)
    if np.min(outer - inner) <= 2e-4:
        raise ValueError("the poppet closes the seat")
    mesh = hex_mesh()
    grid = _wedge_grid(x, inner, outer, level.cells(30))
    middle = 0.5 * (x[1:] + x[:-1])
    core = np.where((middle > tip) & (middle < 0.085), "plug", "axis").astype(object)
    mesh.add_block(
        grid,
        {
            "i0": "inlet", "i1": "outlet", "j0": core[:, None], "j1": "walls",
            "k0": "wedge_back", "k1": "wedge_front",
        },
    )  # fmt: skip
    rate = 2.0e-4
    patches = _finish(
        case,
        mesh,
        {
            "inlet": flow_inlet(rate / WEDGES), "outlet": pressure_outlet(), "walls": wall(),
            "plug": wall(), "axis": slip(), "wedge_back": wedge(), "wedge_front": wedge(),
        },
    )  # fmt: skip
    functions = post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet")
    functions |= post.forces("plug", ["plug"], WATER["density"])
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam",
            velocity=rate / (pi * pipe_radius**2),
            length=0.004,
            iterations=level.iterations,
            second_order=level.resolution > 1,
            functions=functions,
            **WATER,
        ),
    )

    def measure(directory):
        loss = post.last_surface(directory, "total_p_inlet") - post.last_surface(
            directory, "total_p_outlet"
        )
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "pressure_loss_pa": WATER["density"] * loss,
            "flow_force_n": abs(float(post.last_force(directory, "plug")[0])) * WEDGES,
        }

    return run_plan(solve=[["simpleFoam"]], measure=measure)


def labyrinth_maze(case, design, level):
    """Water driven into an air-filled annular gap sealed by square teeth."""
    shaft, bore = 0.02, 0.0235
    teeth = round(design["tooth_count"])
    clearance, length = design["clearance_m"], design["length_m"]
    x = np.linspace(0.0, length, level.cells(200) + 1)
    radial = level.cells(18, 10)
    mesh = hex_mesh()
    # teeth rise from the shaft over the middle 80 percent of the length
    phase = np.clip((x - 0.1 * length) / (0.8 * length), 0, 1) * teeth
    ramp = 0.12
    local = np.mod(phase, 1.0)
    shape = np.clip(np.minimum((local - 0.25) / ramp, (0.75 - local) / ramp), 0, 1)
    shape[(x <= 0.1 * length) | (x >= 0.9 * length)] = 0.0
    inner = shaft + (bore - clearance - shaft) * shape
    mesh.add_block(
        _wedge_grid(x, inner, np.full(len(x), bore), radial),
        {
            "i0": "inlet", "i1": "outlet", "j0": "shaft", "j1": "walls",
            "k0": "wedge_back", "k1": "wedge_front",
        },
    )  # fmt: skip
    pressure = 20.0  # m2/s2, so a 20 kPa water head
    patches = _finish(
        case,
        mesh,
        {
            "inlet": total_pressure_inlet(pressure * 1000.0, water=1.0),
            "outlet": pressure_outlet(), "shaft": wall(), "walls": wall(),
            "wedge_back": wedge(), "wedge_front": wedge(),
        },
    )  # fmt: skip
    functions = post.flux("inlet", "outlet")
    write_flow_case(
        case,
        patches,
        flow(
            "interFoam",
            velocity=5.0,
            length=clearance,
            turbulent=False,
            time_step=2e-6,
            time_steps=level.time_steps,
            gravity=(0.0, 0.0, 0.0),
            functions=functions,
        ),
    )

    def measure(directory):
        return {
            "flow_mismatch": post.imbalance(directory, "inlet", "outlet"),
            "leakage_m3_s": abs(post.last_flux(directory, "outlet")) * WEDGES,
            "inflow_m3_s": abs(post.last_flux(directory, "inlet")) * WEDGES,
            "length_m": length,
        }

    return run_plan(solve=[["interFoam"]], measure=measure)


def human_comfort(case, design, level):
    """A 4 x 3 x 2.7 m room with one seated heat source, supply low and exhaust high."""
    lower, upper = (0.0, 0.0, 0.0), (4.0, 3.0, 2.7)
    cells = [level.cells(n) for n in (40, 30, 27)]
    grid = box_grid(lower, upper, cells)
    centre = 0.25 * (grid[1:, 1:, 1:] + grid[:-1, :-1, 1:] + grid[1:, :-1, :-1]
                     + grid[:-1, 1:, :-1])  # fmt: skip
    x, y, z = np.moveaxis(centre, -1, 0)
    person = (np.abs(x - 2.4) < 0.2) & (np.abs(y - 1.5) < 0.25) & (z < 1.2)
    occupied = (x > 1.0) & (x < 3.4) & (y > 0.5) & (y < 2.5) & (z < 1.8)
    zone = np.where(occupied & ~person, "occupied", None).astype(object)
    # cell-face centres of the two end walls
    face_y = 0.5 * (grid[0, 1:, 0, 1] + grid[0, :-1, 0, 1])
    face_z = 0.5 * (grid[0, 0, 1:, 2] + grid[0, 0, :-1, 2])
    across = np.abs(face_y - 1.5)[:, None] < 0.5
    vent = design["vent_height_m"]
    supply = np.where(across & (np.abs(face_z - vent) < 0.15)[None, :], "inlet", "walls")
    exhaust = np.where(across & (np.abs(face_z - 2.5) < 0.12)[None, :], "outlet", "walls")
    mesh = hex_mesh()
    mesh.add_block(
        grid,
        {"i0": supply.astype(object), "i1": exhaust.astype(object), "k0": "floor", "k1": "walls",
         "j0": "walls", "j1": "walls"},
        keep=~person,
        exposed="person",
        zone=zone,
    )  # fmt: skip
    speed = design["inlet_speed_m_s"]
    supply_temperature = design["supply_temperature_k"]
    patches = _finish(
        case,
        mesh,
        {
            "inlet": velocity_inlet((speed, 0, 0), temperature=supply_temperature),
            "outlet": pressure_outlet(101325.0), "walls": wall(), "floor": wall(296.0),
            "person": wall(307.0),
        },
    )  # fmt: skip
    functions = post.flux("inlet", "outlet")
    functions |= {
        "speed": {"type": "mag", "libs": "(fieldFunctionObjects)", "field": "U",
                  "result": "magU", "writeControl": "none", "log": False}
    }  # fmt: skip
    functions |= post.volume_value("occupied_temperature", "volAverage", "T", "occupied")
    functions |= post.volume_value("occupied_speed", "volAverage", "magU", "occupied")
    write_flow_case(
        case,
        patches,
        flow(
            "buoyantSimpleFoam",
            velocity=speed,
            length=0.3,
            iterations=level.iterations,
            temperature=295.0,
            functions=functions,
        ),
    )

    def measure(directory):
        temperature = post.last_volume(directory, "occupied_temperature")
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "occupied_temperature_k": temperature,
            "comfort_deviation_k": abs(temperature - 295.65),
            "draught_speed_m_s": post.last_volume(directory, "occupied_speed"),
        }

    return run_plan(solve=[["buoyantSimpleFoam"]], measure=measure)


def _polar_grid(r, theta, z) -> np.ndarray:
    return lattice(
        r, theta, z, lambda a, b, c: (a * np.cos(b), a * np.sin(b), c)
    )


def rushton_impeller(case, design, level):
    """An unbaffled 800 mm vessel with a six-blade disc turbine on a central shaft."""
    vessel, height, shaft = 0.4, 0.9, 0.02
    radius, clearance = design["impeller_diameter_m"] / 2, design["impeller_height_m"]
    omega = design["speed_rad_s"]
    blade_inner, blade_height, thickness = 0.62 * radius, 0.4 * radius, 0.012
    r = np.unique(
        np.concatenate(
            [
                np.linspace(shaft, blade_inner, level.cells(6) + 1),
                np.linspace(blade_inner, radius, level.cells(8) + 1),
                np.linspace(radius, vessel, level.cells(14) + 1),
            ]
        )
    )
    z = np.unique(
        np.concatenate(
            [
                np.linspace(0, clearance - blade_height / 2, level.cells(8) + 1),
                np.linspace(clearance - blade_height / 2, clearance - thickness / 2,
                            level.cells(4) + 1),
                np.linspace(clearance + thickness / 2, clearance + blade_height / 2,
                            level.cells(4) + 1),
                np.linspace(clearance + blade_height / 2, height, level.cells(16) + 1),
            ]
        )
    )  # fmt: skip
    per_blade = level.cells(12, 8)
    theta = np.linspace(0, 2 * pi, 6 * per_blade + 1)
    rc, zc = 0.5 * (r[1:] + r[:-1]), 0.5 * (z[1:] + z[:-1])
    shape = (len(rc), len(theta) - 1, len(zc))
    rr = np.broadcast_to(rc[:, None, None], shape)
    zz = np.broadcast_to(zc[None, None, :], shape)
    sector = np.broadcast_to((np.arange(len(theta) - 1) % per_blade)[None, :, None], shape)
    disc = (np.abs(zz - clearance) < thickness / 2) & (rr < 0.78 * radius)
    blades = (
        (sector == 0)
        & (rr > blade_inner)
        & (rr < radius)
        & (np.abs(zz - clearance) < blade_height / 2)
    )
    rotating = (rr < radius + 0.04) & (np.abs(zz - clearance) < blade_height / 2 + 0.04)
    mesh = hex_mesh()
    mesh.add_block(
        _polar_grid(r, theta, z),
        {"i0": "shaft", "i1": "tank", "k0": "bottom", "k1": "lid"},
        keep=~(disc | blades),
        exposed="impeller",
        zone=np.where(rotating, "rotor", None).astype(object),
    )
    patches = _finish(
        case,
        mesh,
        {
            "shaft": rotating_wall(omega), "tank": wall(), "bottom": wall(), "lid": slip(),
            "impeller": wall(),
        },
    )  # fmt: skip
    functions = post.forces("impeller", ["impeller"], WATER["density"])
    functions |= {
        "speed": {"type": "mag", "libs": "(fieldFunctionObjects)", "field": "U",
                  "result": "magU", "writeControl": "none", "log": False}
    }  # fmt: skip
    functions |= post.volume_value("mean_speed", "volAverage", "magU")
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam",
            velocity=omega * radius,
            length=blade_height,
            iterations=level.iterations,
            closed=True,
            rotations=(rotation("rotor", omega, ("tank", "bottom", "lid")),),
            functions=functions,
            **WATER,
        ),
    )

    def measure(directory):
        torque = float(post.last_moment(directory, "impeller")[2])
        return {
            "shaft_power_w": -torque * omega,
            "mean_speed_m_s": post.last_volume(directory, "mean_speed"),
        }

    return run_plan(solve=[["simpleFoam"]], measure=measure)


def mechanical_siren(case, design, level):
    """A slotted rotor chopping the flow just ahead of a slotted stator."""
    hub, tip = 0.03, 0.08
    ports = round(design["port_count"])
    gap, omega = design["gap_m"], design["speed_rad_s"]
    rotor_start, thickness, lead = 0.1, 0.008, 0.003
    rotor_face = rotor_start - lead  # sliding faces stay clear of the solid discs
    rotor_end = rotor_start + thickness + gap / 2
    stator_start = rotor_start + thickness + gap
    per_port = max(8, level.cells(96) // ports)
    theta = np.linspace(0, 2 * pi, ports * per_port + 1)
    r = np.linspace(hub, tip, level.cells(10) + 1)
    open_slot = (np.arange(len(theta) - 1) % per_port) < per_port // 2

    def block(z_nodes, solid_range):
        zc = 0.5 * (z_nodes[1:] + z_nodes[:-1])
        solid = (zc > solid_range[0]) & (zc < solid_range[1])
        covered = (r[:-1] >= hub + 0.25 * (tip - hub)) & (r[1:] <= tip - 0.1 * (tip - hub))
        mask = covered[:, None, None] & ~open_slot[None, :, None] & solid[None, None, :]
        return _polar_grid(r, theta, z_nodes), ~mask

    def axial(*breaks):
        parts = [np.linspace(a, b, level.cells(n) + 1) for a, b, n in breaks]
        return np.unique(np.concatenate(parts))

    pieces = (
        (axial((0.0, rotor_face, 16)), (-1.0, -1.0), 0, {"k0": "inlet", "k1": "front_static"}),
        (
            axial((rotor_face, rotor_start, 2),
                  (rotor_start, rotor_start + thickness, 4),
                  (rotor_start + thickness, rotor_end, 2)),
            (rotor_start, rotor_start + thickness), 1,
            {"k0": "front_rotor", "k1": "rear_rotor"},
        ),
        (
            axial((rotor_end, stator_start, 2),
                  (stator_start, stator_start + thickness, 4),
                  (stator_start + thickness, 0.25, 18)),
            (stator_start, stator_start + thickness), 0,
            {"k0": "rear_static", "k1": "outlet"},
        ),
    )  # fmt: skip
    mesh = hex_mesh()
    for z_nodes, solid_range, region, caps in pieces:
        points, keep = block(z_nodes, solid_range)
        walls = "rotor_walls" if region else "walls"
        mesh.add_block(
            points, {"i0": walls, "i1": walls, **caps}, keep=keep, exposed=walls, region=region,
            zone="rotor" if region else None,
        )  # fmt: skip
    rate = 0.08
    patches = _finish(
        case,
        mesh,
        {
            "inlet": flow_inlet(rate), "outlet": pressure_outlet(), "walls": wall(),
            "rotor_walls": moving_wall(),
            "front_static": interface("front_rotor"), "front_rotor": interface("front_static"),
            "rear_static": interface("rear_rotor"), "rear_rotor": interface("rear_static"),
        },
    )  # fmt: skip
    radius = 0.7 * tip
    functions = post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet")
    functions |= post.probes("pressure_probes", ((radius, 0.003, 0.18), (0.003, radius, 0.05)))
    step = ceil(1e6 * radians(1.5) / omega) * 1e-6
    write_flow_case(
        case,
        patches,
        flow(
            "pimpleFoam",
            velocity=rate / (pi * (tip**2 - hub**2)),
            length=0.02,
            time_step=step,
            time_steps=level.time_steps,
            motion={
                "dynamicFvMesh": "dynamicMotionSolverFvMesh",
                "motionSolverLibs": ("fvMotionSolvers",),
                "motionSolver": "solidBody",
                "cellZone": "rotor",
                "solidBodyMotionFunction": "rotatingMotion",
                "rotatingMotionCoeffs": {"origin": (0, 0, 0), "axis": (0, 0, 1), "omega": omega},
            },
            functions=functions,
        ),
    )

    def measure(directory):
        drop = post.last_surface(directory, "total_p_inlet") - post.last_surface(
            directory, "total_p_outlet"
        )
        return {
            "flow_mismatch": post.imbalance(directory, "inlet", "outlet"),
            "pressure_amplitude_pa": 1.2 * post.probe_fluctuation(directory, "pressure_probes"),
            "pressure_drop_pa": 1.2 * drop,
        }

    return run_plan(solve=[["pimpleFoam"]], measure=measure)


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    return cfd_case(
        name, "internal", title, "structured", solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip


CASES = [
    _case(
        "pipe", "Water pipe with a 180 degree bend", pipe,
        (variable("diameter_m", 0.08, 0.12, 0.1), variable("bend_radius_m", 0.15, 0.4, 0.25)),
        "pressure_loss_pa", (at_most("volume_m3"),), positive=("pressure_loss_pa",),
        simplification="flow only, no structural analysis",
    ),
    _case(
        "parametric_pipe", "Bend with nine free diameters", parametric_pipe,
        tuple(variable(n, 0.085, 0.12, 0.1) for n in STATIONS),
        "pressure_loss_pa", (at_most("volume_m3", 0.05),), positive=("pressure_loss_pa",),
    ),
    _case(
        "hydraulic_valve", "Axisymmetric poppet valve", hydraulic_valve,
        (
            variable("cone_half_angle_deg", 30.0, 55.0, 42.0),
            variable("lift_m", 0.0012, 0.003, 0.002),
            variable("seat_radius_m", 0.0055, 0.0072, 0.0064),
        ),
        "pressure_loss_pa", (at_most("flow_force_n", 0.15),), positive=("pressure_loss_pa",),
        simplification="axisymmetric, no structural analysis",
    ),
    _case(
        "labyrinth_maze", "Labyrinth seal filling with water", labyrinth_maze,
        (
            variable("tooth_count", 3.0, 7.0, 5.0),
            variable("clearance_m", 0.0004, 0.0012, 0.0007),
            variable("length_m", 0.04, 0.07, 0.05),
        ),
        "leakage_m3_s", (at_most("length_m", 0.1),), solver="interFoam",
        simplification="axisymmetric, short transient from an air-filled gap",
    ),
    _case(
        "human_comfort", "Ventilated room with a seated occupant", human_comfort,
        (
            variable("inlet_speed_m_s", 0.15, 0.6, 0.3),
            variable("supply_temperature_k", 288.0, 296.0, 292.0),
            variable("vent_height_m", 0.3, 1.8, 0.5),
        ),
        "comfort_deviation_k", (at_most("draught_speed_m_s", 0.3),), solver="buoyantSimpleFoam",
        simplification="box manikin, no radiation or humidity",
    ),
    _case(
        "rushton_impeller", "Rushton turbine in a stirred vessel", rushton_impeller,
        (
            variable("impeller_diameter_m", 0.22, 0.32, 0.267),
            variable("impeller_height_m", 0.2, 0.4, 0.27),
            variable("speed_rad_s", 7.0, 16.0, 10.5),
        ),
        "shaft_power_w", (at_least("mean_speed_m_s"),),
        positive=("shaft_power_w",), simplification="single phase, flat bottom",
    ),
    _case(
        "mechanical_siren", "Sliding-mesh siren", mechanical_siren,
        (
            variable("port_count", 4.0, 10.0, 6.0),
            variable("gap_m", 0.002, 0.008, 0.004),
            variable("speed_rad_s", 200.0, 500.0, 314.0),
        ),
        "pressure_amplitude_pa", (at_most("pressure_drop_pa", 0.5),), maximize=True,
        solver="pimpleFoam", simplification="probe pressures only, short transient",
    ),
]
