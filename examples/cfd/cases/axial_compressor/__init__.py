"""Subsonic axial compressor stage."""

from foam.case import at_least, variable
from foam.passage import axial_channel, blade_row, machine

from ..axial import _EQUAL, _annulus, _case, _fan_row
from ..turbo import build_machine, duty


def axial_compressor(case, design, level):
    hub, tip, speed = 0.15, 0.2, 70.0
    omega = 550.0 * design["speed_ratio"]
    flow = 1.17 * speed * _annulus(hub, tip)
    rotor = _fan_row("rotor", 0.2, 0.22, hub, tip, omega, speed, 2.0, design["camber_deg"])
    stator_in = design["stator_angle_deg"]
    stator = blade_row("stator", 0.58, 0.78, (stator_in, stator_in - 5), (0.0, 0.0))
    wheel = machine(*axial_channel(0.2, hub, tip), blades=18, rows=(rotor, stator))
    point = duty(
        "pump", speed, 0.045, flow=flow, compressible=True, inlet_pressure=101325.0,
        outlet_pressure=101325.0,
    )  # fmt: skip
    return build_machine(case, wheel, point, level)


CASE = _case(
    "axial_compressor",
    "Subsonic axial compressor stage",
    axial_compressor,
    (
        variable("camber_deg", 4.0, 14.0, 8.0),
        variable("stator_angle_deg", 20.0, 45.0, 32.0),
        variable("speed_ratio", 0.8, 1.1, 1.0),
    ),
    "efficiency",
    (at_least("pressure_ratio", 0.01),),
    maximize=True,
    positive=("shaft_power_w",),
    solver="rhoSimpleFoam",
    simplification=_EQUAL,
)
