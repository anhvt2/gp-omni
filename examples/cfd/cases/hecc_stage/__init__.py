"""Compressor stage with a vaned diffuser."""

from math import pi

from foam.case import at_least, variable
from foam.passage import blade_row, machine, radial_channel

from ..radial import _case
from ..turbo import build_machine, duty, relative_angle


def hecc_stage(case, design, level):
    outer, omega, speed = 0.09, 2200.0 * design["speed_ratio"], 70.0
    hub, shroud = radial_channel(0.02, 0.05, outer, 0.009, 0.035, inlet=0.03, outlet=0.07)
    inlet_angle = relative_angle(omega, 0.039, speed)
    impeller = blade_row(
        "rotor", 0.2, 0.55, (inlet_angle,) * 2, (-35.0,) * 2, thickness=0.06, omega=omega
    )
    angle = design["diffuser_angle_deg"]
    diffuser = blade_row("diffuser", 0.66, 0.88, (angle,) * 2, (angle - 10,) * 2, thickness=0.05)
    wheel = machine(hub, shroud, blades=15, rows=(impeller, diffuser))
    flow = 1.17 * speed * pi * (0.05**2 - 0.02**2)
    point = duty("pump", speed, 0.03, flow=flow, compressible=True)
    return build_machine(case, wheel, point, level)


CASE = _case(
    "hecc_stage",
    "Compressor stage with a vaned diffuser",
    hecc_stage,
    (
        variable("speed_ratio", 0.8, 1.1, 1.0),
        variable("diffuser_angle_deg", 50.0, 66.0, 60.0),
    ),
    "efficiency",
    (at_least("pressure_ratio", 0.01),),
    maximize=True,
    positive=("shaft_power_w",),
    solver="rhoSimpleFoam",
    simplification="representative stage with equal impeller and diffuser blade counts",
)
