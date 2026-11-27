"""Axial turbomachinery: fans, pumps, turbines, propellers and a compressor.

Rotors turn about +z and the flow runs along +z. Open rotors are modelled with an axial
inflow and a slip far field; the blade angle limit of the mesher keeps their tip speed
near 2.5 times the inflow speed.
"""

from math import atan, degrees, pi, radians, tan

from foam.case import at_least, at_most, cfd_case, variable
from foam.passage import axial_channel, blade_row, machine

from .turbo import build_machine, duty, relative_angle


def _annulus(hub: float, tip: float) -> float:
    return pi * (tip**2 - hub**2)


def _fan_row(name, start, chord, hub, tip, omega, speed, offset, turn) -> blade_row:
    """A rotor row aligned with its relative inflow and turning it toward axial."""
    angles = [relative_angle(omega, r, speed) + offset for r in (hub, tip)]
    # the turning acts against the sense of the relative swirl
    sense = 1 if omega > 0 else -1
    return blade_row(
        name,
        start,
        start + chord,
        (angles[0], angles[1]),
        (angles[0] + sense * turn, angles[1] + sense * 0.6 * turn),
        omega=omega,
    )


def _fan(case, level, camber, offset, chord, blades, point_kwargs=None, sliding=False,
         speed_ratio=1.0):  # fmt: skip
    hub, tip, omega, speed = 0.125, 0.25, 150.0 * speed_ratio, 15.0
    row = _fan_row("rotor", 0.35, chord, hub, tip, omega, speed, offset, camber)
    wheel = machine(
        *axial_channel(0.3, hub, tip),
        blades=blades,
        rows=(row,),
        full_wheel=sliding,
        sliding=sliding,
    )
    point = duty(
        "pump", speed, chord * 0.3, flow=speed * _annulus(hub, tip), **(point_kwargs or {})
    )
    return build_machine(case, wheel, point, level)


def axial_fan(case, design, level):
    return _fan(
        case, level, design["camber_deg"], design["inlet_offset_deg"], design["chord_fraction"], 9
    )


def axial_fan_optimization(case, design, level):
    return _fan(
        case,
        level,
        design["camber_deg"],
        design["inlet_offset_deg"],
        0.25,
        round(design["blade_count"]),
    )


def axial_fan_acoustic(case, design, level):
    probes = ((0.19, 0.01, 0.03), (0.01, 0.19, 0.27))
    kwargs = {"transient": True, "time_step": 2.3e-4, "probes": probes}
    return _fan(
        case, level, 10.0, 2.0, 0.25, round(design["blade_count"]), kwargs, sliding=True,
        speed_ratio=design["speed_ratio"],
    )  # fmt: skip


def double_rotor_fan(case, design, level):
    hub, tip, omega, speed = 0.125, 0.25, 110.0, 15.0
    ratio = design["speed_ratio"]
    front = _fan_row(
        "front", 0.2, 0.18, hub, tip, omega, speed, 2.0, design["front_camber_deg"]
    )
    # the rear rotor counter-rotates and meets the swirl left by the front rotor
    rear_in = [degrees(atan((ratio * omega + 0.3 * omega) * r / speed)) - 2 for r in (hub, tip)]
    turn = design["rear_camber_deg"]
    rear = blade_row(
        "rear", 0.6, 0.78, (rear_in[0], rear_in[1]),
        (rear_in[0] - turn, rear_in[1] - 0.6 * turn), omega=-ratio * omega,
    )  # fmt: skip
    wheel = machine(*axial_channel(0.4, hub, tip), blades=9, rows=(front, rear))
    point = duty("pump", speed, 0.07, flow=speed * _annulus(hub, tip))
    return build_machine(case, wheel, point, level)


def _stage_pump(case, level, hub, tip, omega, speed, offset, turn, stator_offset, blades,
                length, **point_kwargs):  # fmt: skip
    """A rotor followed by a straightening stator with the same blade count."""
    rotor = _fan_row("rotor", 0.2, 0.24, hub, tip, omega, speed, offset, turn)
    swirl_angles = []
    for radius, outlet in zip((hub, tip), rotor.outlet_angle):
        swirl = omega * radius + speed * tan(radians(max(-68.0, outlet)))
        swirl_angles.append(degrees(atan(max(swirl, 0.0) / speed)) + stator_offset)
    stator = blade_row("stator", 0.58, 0.8, tuple(swirl_angles), (0.0, 0.0))
    wheel = machine(*axial_channel(length, hub, tip), blades=blades, rows=(rotor, stator))
    point = duty(
        "pump", speed, 0.24 * length, flow=speed * _annulus(hub, tip), **point_kwargs
    )
    return build_machine(case, wheel, point, level)


def axial_pump(case, design, level):
    return _stage_pump(
        case, level, 0.06, 0.12, 120.0, 6.5, design["incidence_deg"], design["camber_deg"],
        design["stator_offset_deg"], 7, 0.2, density=1000.0, viscosity=1e-6,
    )  # fmt: skip


def parametric_axial_blood_pump(case, design, level):
    tip = 0.008
    return _stage_pump(
        case, level, design["hub_ratio"] * tip, tip, design["speed_rad_s"], 1.4,
        design["pitch_deg"], 12.0, 0.0, 5, 0.03, density=1055.0, viscosity=3.3e-6,
        wall_shear=True,
    )  # fmt: skip


def _turbine(case, level, hub, tip, omega, speed, vane_angle, exit_offset, blades, length,
             **point_kwargs):  # fmt: skip
    """Nozzle vanes that swirl the flow, followed by a rotor that removes the swirl."""
    vane = blade_row("vane", 0.2, 0.38, (0.0, 0.0), (vane_angle, vane_angle - 6))
    swirl = speed * tan(radians(vane_angle - 3))
    radii = (hub, tip)
    inlet = tuple(degrees(atan((swirl - omega * r) / speed)) for r in radii)
    outlet = tuple(relative_angle(omega, r, speed) + exit_offset for r in radii)
    rotor = blade_row("rotor", 0.55, 0.75, inlet, outlet, omega=omega)
    wheel = machine(*axial_channel(length, hub, tip), blades=blades, rows=(vane, rotor))
    point = duty(
        "turbine", speed, 0.2 * length, flow=speed * _annulus(hub, tip), **point_kwargs
    )
    return build_machine(case, wheel, point, level)


def axial_turbine(case, design, level):
    return _turbine(
        case, level, 0.2, 0.3, 120.0 * design["speed_ratio"], 20.0, design["vane_angle_deg"],
        design["exit_offset_deg"], 26, 0.25,
    )  # fmt: skip


def kaplan_turbine(case, design, level):
    return _turbine(
        case, level, 0.1, 0.22, 35.0 * design["speed_ratio"], 5.0, design["guide_angle_deg"],
        design["runner_offset_deg"], 11, 0.35, density=1000.0, viscosity=1e-6,
    )  # fmt: skip


def _open_rotor(
    case, level, kind, radius, hub, blades, speed, omega, incidence, chord_fraction,
    density=1.2, viscosity=1.5e-5,
):  # fmt: skip
    """One passage of an unducted rotor of tip ``radius`` in a far field 3 radii wide."""
    far = 3.0 * radius
    length = 2.4 * radius
    sense = 1.0 if kind == "propeller" else -1.0

    def angle(height: float, extra: float = 0.0) -> float:
        local = hub + height * (radius - hub)
        return relative_angle(omega, local, speed) + sense * incidence + extra

    row = blade_row(
        "rotor",
        0.38,
        0.38 + chord_fraction,
        angle,
        lambda height: angle(height, sense * 2.0),
        thickness=0.04,
        omega=omega,
        span=(radius - hub) / (far - hub),
    )
    wheel = machine(
        *axial_channel(length, hub, far), blades=blades, rows=(row,), outer="farfield"
    )
    reference = radius if kind == "wind" else chord_fraction * length
    point = duty(kind, speed, reference, density=density, viscosity=viscosity)
    return build_machine(case, wheel, point, level)


def ship_hull_propeller(case, design, level):
    radius, omega = 0.125, 50.0 * design["speed_ratio"]
    return _open_rotor(
        case, level, "propeller", radius, 0.025, 4, 2.5, omega, design["pitch_deg"],
        design["chord_fraction"], 1000.0, 1e-6,
    )  # fmt: skip


def potsdam_propeller(case, design, level):
    return _open_rotor(
        case, level, "propeller", 0.125, 0.0375, 5, design["inflow_m_s"], 94.25,
        design["pitch_deg"], 0.14, 1000.0, 1e-6,
    )  # fmt: skip


def propeller_dji_9450(case, design, level):
    radius, omega = 0.12, 560.0 * design["speed_ratio"]
    height_twist = design["tip_pitch_deg"] - design["root_pitch_deg"]
    speed = 560.0 * radius / 2.5
    far = 3.0 * radius
    hub = 0.015

    def angle(height: float, extra: float = 0.0) -> float:
        local = hub + height * (radius - hub)
        pitch = design["root_pitch_deg"] + height * height_twist
        return relative_angle(omega, local, speed) + pitch + extra

    row = blade_row(
        "rotor", 0.4, 0.52, angle, lambda h: angle(h, 2.0), thickness=0.03, omega=omega,
        span=(radius - hub) / (far - hub),
    )  # fmt: skip
    wheel = machine(*axial_channel(0.29, hub, far), blades=2, rows=(row,), outer="farfield")
    point = duty("propeller", speed, 0.03)
    return build_machine(case, wheel, point, level)


def propeller_optimization(case, design, level):
    radius = design["diameter_m"] / 2
    omega = 500.0
    return _open_rotor(
        case, level, "propeller", radius, 0.12 * radius, 3, 500.0 * 0.12 / 2.5, omega,
        design["pitch_deg"], design["chord_fraction"],
    )  # fmt: skip


def wind_turbine(case, design, level):
    radius, hub, speed = 5.0, 0.6, 8.0
    omega = design["tip_speed_ratio"] * speed / radius
    far = 3.0 * radius

    def angle(height: float, extra: float = 0.0) -> float:
        local = hub + height * (radius - hub)
        pitch = design["pitch_deg"] + design["twist_deg"] * (1 - height)
        return relative_angle(omega, local, speed) - pitch + extra

    row = blade_row(
        "rotor", 0.4, 0.5, angle, lambda h: angle(h, -2.0), thickness=0.04, omega=omega,
        span=(radius - hub) / (far - hub),
    )  # fmt: skip
    wheel = machine(*axial_channel(12.0, hub, far), blades=3, rows=(row,), outer="farfield")
    point = duty("wind", speed, radius)
    return build_machine(case, wheel, point, level)


def axial_compressor(case, design, level):
    hub, tip, speed = 0.15, 0.2, 70.0
    omega = 550.0 * design["speed_ratio"]
    flow = 1.17 * speed * _annulus(hub, tip)
    rotor = _fan_row(
        "rotor", 0.2, 0.22, hub, tip, omega, speed, 2.0, design["camber_deg"]
    )
    stator_in = design["stator_angle_deg"]
    stator = blade_row("stator", 0.58, 0.78, (stator_in, stator_in - 5), (0.0, 0.0))
    wheel = machine(*axial_channel(0.2, hub, tip), blades=18, rows=(rotor, stator))
    point = duty(
        "pump", speed, 0.045, flow=flow, compressible=True, inlet_pressure=101325.0,
        outlet_pressure=101325.0,
    )  # fmt: skip
    return build_machine(case, wheel, point, level)


def _case(name, title, build, variables, objective, constraints, **options):
    options.setdefault("solver", "simpleFoam")
    solver = options.pop("solver")
    return cfd_case(
        name, "axial_turbo", title, "structured", solver, variables, objective, constraints,
        build, **options,
    )  # fmt: skip


_EQUAL = "equal blade counts, frozen rotor, no tip clearance"
_OPEN = "axial inflow, slip far field, low tip speed ratio set by the mesher's angle limit"
_PUMPING = ("pressure_rise_pa", "shaft_power_w")
_PUSHING = ("thrust_n", "shaft_power_w")

CASES = [
    _case(
        "axial_fan", "Ducted axial fan, one blade passage", axial_fan,
        (
            variable("inlet_offset_deg", -3.0, 6.0, 2.0),
            variable("camber_deg", 4.0, 18.0, 10.0),
            variable("chord_fraction", 0.18, 0.32, 0.25),
        ),
        "efficiency", (at_least("pressure_rise_pa"),), maximize=True, positive=_PUMPING,
        simplification="shrouded blades without tip clearance",
    ),
    _case(
        "axial_fan_optimization", "Axial fan with a free blade count", axial_fan_optimization,
        (
            variable("inlet_offset_deg", -3.0, 6.0, 2.0),
            variable("camber_deg", 4.0, 18.0, 10.0),
            variable("blade_count", 6.0, 12.0, 9.0),
        ),
        "efficiency", (at_least("pressure_rise_pa"),), maximize=True, positive=_PUMPING,
        simplification="shrouded blades without tip clearance",
    ),
    _case(
        "axial_fan_acoustic", "Sliding-mesh axial fan with pressure probes", axial_fan_acoustic,
        (variable("speed_ratio", 0.8, 1.15, 1.0), variable("blade_count", 5.0, 9.0, 7.0)),
        "pressure_fluctuation_pa", (at_least("pressure_rise_pa", 0.3),), solver="pimpleFoam",
        simplification="probe pressures only, no far-field acoustics",
    ),
    _case(
        "double_rotor_fan", "Counter-rotating fan", double_rotor_fan,
        (
            variable("front_camber_deg", 4.0, 16.0, 9.0),
            variable("rear_camber_deg", 4.0, 16.0, 9.0),
            variable("speed_ratio", 0.7, 1.2, 1.0),
        ),
        "efficiency", (at_least("pressure_rise_pa"),), maximize=True, positive=_PUMPING,
        simplification=_EQUAL,
    ),
    _case(
        "axial_pump", "Axial water pump with stator", axial_pump,
        (
            variable("incidence_deg", -2.0, 6.0, 2.0),
            variable("camber_deg", 5.0, 16.0, 10.0),
            variable("stator_offset_deg", -6.0, 6.0, 0.0),
        ),
        "efficiency", (at_least("pressure_rise_pa"),), maximize=True, positive=_PUMPING,
        simplification=_EQUAL,
    ),
    _case(
        "parametric_axial_blood_pump", "Miniature axial blood pump", parametric_axial_blood_pump,
        (
            variable("pitch_deg", 0.0, 8.0, 4.0),
            variable("hub_ratio", 0.4, 0.62, 0.5),
            variable("speed_rad_s", 500.0, 800.0, 650.0),
        ),
        "wall_shear_pa", (at_least("pressure_rise_pa", 0.15),), positive=_PUMPING,
        simplification=_EQUAL + ", wall shear in place of a blood damage model",
    ),
    _case(
        "axial_turbine", "Axial turbine stage", axial_turbine,
        (
            variable("vane_angle_deg", 50.0, 68.0, 60.0),
            variable("exit_offset_deg", -6.0, 14.0, 4.0),
            variable("speed_ratio", 0.75, 1.2, 1.0),
        ),
        "efficiency", (at_least("output_power_w"),), maximize=True,
        positive=("output_power_w", "pressure_drop_pa"),
        simplification=_EQUAL + ", no spiral casing",
    ),
    _case(
        "kaplan_turbine", "Kaplan runner behind guide vanes", kaplan_turbine,
        (
            variable("guide_angle_deg", 40.0, 62.0, 52.0),
            variable("runner_offset_deg", -6.0, 14.0, 4.0),
            variable("speed_ratio", 0.75, 1.2, 1.0),
        ),
        "efficiency", (at_least("output_power_w"),), maximize=True,
        positive=("output_power_w", "pressure_drop_pa"),
        simplification=_EQUAL + ", axial guide vanes, no spiral or draft tube",
    ),
    _case(
        "ship_hull_propeller", "Marine propeller in open water", ship_hull_propeller,
        (
            variable("pitch_deg", 2.0, 10.0, 5.0),
            variable("chord_fraction", 0.1, 0.18, 0.14),
            variable("speed_ratio", 0.85, 1.1, 1.0),
        ),
        "efficiency", (at_least("thrust_n"),), maximize=True, positive=_PUSHING,
        simplification=_OPEN + ", no hull",
    ),
    _case(
        "potsdam_propeller", "Five-blade propeller advance sweep", potsdam_propeller,
        (variable("pitch_deg", 2.0, 10.0, 5.0), variable("inflow_m_s", 4.0, 6.0, 4.7)),
        "efficiency", (at_least("thrust_n"),), maximize=True, positive=_PUSHING,
        simplification=_OPEN + ", own blade shape, no cavitation",
    ),
    _case(
        "propeller_dji_9450", "Two-blade drone propeller", propeller_dji_9450,
        (
            variable("root_pitch_deg", 3.0, 12.0, 7.0),
            variable("tip_pitch_deg", 1.0, 8.0, 4.0),
            variable("speed_ratio", 0.85, 1.1, 1.0),
        ),
        "shaft_power_w", (at_least("thrust_n"),), positive=_PUSHING,
        simplification=_OPEN + ", own blade shape, no acoustics",
    ),
    _case(
        "propeller_optimization", "Three-blade propeller sizing", propeller_optimization,
        (
            variable("pitch_deg", 2.0, 10.0, 5.0),
            variable("chord_fraction", 0.08, 0.16, 0.12),
            variable("diameter_m", 0.2, 0.28, 0.24),
        ),
        "shaft_power_w", (at_least("thrust_n"),), positive=_PUSHING,
        simplification=_OPEN + ", fixed axial inflow in place of static hover",
    ),
    _case(
        "wind_turbine", "Three-blade wind turbine", wind_turbine,
        (
            variable("pitch_deg", 2.0, 12.0, 6.0),
            variable("twist_deg", 0.0, 10.0, 4.0),
            variable("tip_speed_ratio", 1.8, 2.6, 2.2),
        ),
        "output_power_w", (at_most("root_moment_nm"),), maximize=True,
        positive=("output_power_w",), simplification=_OPEN + ", no tower",
    ),
    _case(
        "axial_compressor", "Subsonic axial compressor stage", axial_compressor,
        (
            variable("camber_deg", 4.0, 14.0, 8.0),
            variable("stator_angle_deg", 20.0, 45.0, 32.0),
            variable("speed_ratio", 0.8, 1.1, 1.0),
        ),
        "efficiency", (at_least("pressure_ratio", 0.01),), maximize=True,
        positive=("shaft_power_w",), solver="rhoSimpleFoam", simplification=_EQUAL,
    ),
]
