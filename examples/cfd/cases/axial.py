"""Axial turbomachinery: fans, pumps, turbines, propellers and a compressor.

Rotors turn about +z and the flow runs along +z. Open rotors are modelled with an axial
inflow and a slip far field; the blade angle limit of the mesher keeps their tip speed
near 2.5 times the inflow speed.
"""

from math import atan, degrees, pi, radians, tan

from foam.case import cfd_case
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
    point = duty("pump", speed, 0.24 * length, flow=speed * _annulus(hub, tip), **point_kwargs)
    return build_machine(case, wheel, point, level)


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
    point = duty("turbine", speed, 0.2 * length, flow=speed * _annulus(hub, tip), **point_kwargs)
    return build_machine(case, wheel, point, level)


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
    wheel = machine(*axial_channel(length, hub, far), blades=blades, rows=(row,), outer="farfield")
    reference = radius if kind == "wind" else chord_fraction * length
    point = duty(kind, speed, reference, density=density, viscosity=viscosity)
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
