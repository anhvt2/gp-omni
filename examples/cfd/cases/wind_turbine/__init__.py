"""Three-blade wind turbine."""

from foam.case import at_most, variable
from foam.passage import axial_channel, blade_row, machine

from ..axial import _OPEN, _case
from ..turbo import build_machine, duty, relative_angle


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


CASE = _case(
    "wind_turbine",
    "Three-blade wind turbine",
    wind_turbine,
    (
        variable("pitch_deg", 2.0, 12.0, 6.0),
        variable("twist_deg", 0.0, 10.0, 4.0),
        variable("tip_speed_ratio", 1.8, 2.6, 2.2),
    ),
    "output_power_w",
    (at_most("root_moment_nm"),),
    maximize=True,
    positive=("output_power_w",),
    simplification=_OPEN + ", no tower",
)
