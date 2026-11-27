"""Two-blade drone propeller."""

from foam.case import at_least, variable
from foam.passage import axial_channel, blade_row, machine

from ..axial import _OPEN, _PUSHING, _case
from ..turbo import build_machine, duty, relative_angle


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


CASE = _case(
    "propeller_dji_9450",
    "Two-blade drone propeller",
    propeller_dji_9450,
    (
        variable("root_pitch_deg", 3.0, 12.0, 7.0),
        variable("tip_pitch_deg", 1.0, 8.0, 4.0),
        variable("speed_ratio", 0.85, 1.1, 1.0),
    ),
    "shaft_power_w",
    (at_least("thrust_n"),),
    positive=_PUSHING,
    simplification=_OPEN + ", own blade shape, no acoustics",
)
