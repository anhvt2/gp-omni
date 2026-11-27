"""Radial and mixed-flow turbomachinery.

Impellers have two-dimensional blades and a rectangular spiral casing meshed together with
the passages. Turbines reverse the meridional channel so that the flow runs inward.
"""

from math import atan, degrees, pi, radians, tan

from foam.case import cfd_case
from foam.passage import blade_row, machine, radial_channel, volute

from .turbo import build_machine, duty, relative_angle


def _impeller(
    case, level, *, eye_hub, eye_tip, outer, width, depth, blades, omega, speed,
    outlet_angle, inlet_offset=0.0, casing_height=None, casing_growth=None, offset=0.0,
    sliding=False, **point_kwargs,
):  # fmt: skip
    """A backswept impeller discharging into a spiral casing."""
    extension = 0.12 * outer
    hub, shroud = radial_channel(
        eye_hub, eye_tip, outer, width, depth, inlet=0.4 * outer, outlet=extension
    )
    eye_radius = 0.5 * (eye_hub + eye_tip) + 0.1 * (outer - eye_tip)
    inlet_angle = relative_angle(omega, eye_radius, speed) + inlet_offset
    row = blade_row(
        "rotor", 0.4, 0.86, (inlet_angle,) * 2, (-outlet_angle,) * 2, thickness=0.08,
        omega=omega,
    )  # fmt: skip
    casing = volute(
        height=casing_height or 0.1 * outer,
        growth=casing_growth or 0.35 * outer,
        offset=offset,
    )
    wheel = machine(
        hub, shroud, blades=blades, rows=(row,), full_wheel=True, spiral=casing,
        sliding=sliding,
    )  # fmt: skip
    flow = speed * pi * (eye_tip**2 - eye_hub**2) * point_kwargs.pop("density_inlet", 1.0)
    point = duty("pump", speed, 0.3 * outer, flow=flow, **point_kwargs)
    return build_machine(case, wheel, point, level)


WATER = {"density": 1000.0, "viscosity": 1e-6}
_WATER_PUMP = {
    "eye_hub": 0.03, "eye_tip": 0.07, "outer": 0.14, "depth": 0.05, "blades": 7,
    "omega": 105.0, "speed": 3.0,
}  # fmt: skip


def _inflow_turbine(
    case, level, *, eye_hub, eye_tip, runner, width, depth, blades, omega, flow, vane_angle,
    exit_offset, density_inlet=1.0, **point_kwargs,
):  # fmt: skip
    """Spiral casing, nozzle vanes and an inward-flow runner discharging axially."""
    ring = 0.4 * runner
    hub, shroud = radial_channel(
        eye_hub, eye_tip, runner, width, depth, inlet=0.45 * runner, outlet=ring
    )
    hub, shroud = hub[::-1], shroud[::-1]  # the flow now runs inward
    volume_flow = flow / density_inlet
    inflow = volume_flow / (2 * pi * runner * width)
    exit_speed = volume_flow / (pi * (eye_tip**2 - eye_hub**2))
    swirl = inflow * tan(radians(vane_angle))
    runner_in = degrees(atan((swirl - omega * runner) / inflow))
    runner_out = relative_angle(omega, 0.5 * (eye_hub + eye_tip), exit_speed) + exit_offset
    vanes = blade_row(
        "vane", 0.06, 0.24, (0.35 * vane_angle,) * 2, (vane_angle,) * 2, thickness=0.06
    )
    rotor = blade_row(
        "rotor", 0.36, 0.75, (runner_in,) * 2, (runner_out,) * 2, thickness=0.06, omega=omega
    )
    wheel = machine(
        hub, shroud, blades=blades, rows=(vanes, rotor), full_wheel=True,
        spiral=volute(0.15 * runner, 0.4 * runner, at="start"),
    )  # fmt: skip
    point = duty("turbine", inflow, 0.2 * runner, flow=flow, **point_kwargs)
    return build_machine(case, wheel, point, level)


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    mesher = options.pop("mesher", "structured")
    return cfd_case(
        name, "radial_turbo", title, mesher, solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip


_CASING = "two-dimensional blades, rectangular spiral casing with a radial slot over half a turn"
_PUMPING = ("pressure_rise_pa", "shaft_power_w")
_TURBINE = ("output_power_w",)
