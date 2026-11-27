"""Counter-rotating fan."""

from math import atan, degrees

from foam.case import at_least, variable
from foam.passage import axial_channel, blade_row, machine

from ..axial import _EQUAL, _PUMPING, _annulus, _case, _fan_row
from ..turbo import build_machine, duty


def double_rotor_fan(case, design, level):
    hub, tip, omega, speed = 0.125, 0.25, 110.0, 15.0
    ratio = design["speed_ratio"]
    front = _fan_row("front", 0.2, 0.18, hub, tip, omega, speed, 2.0, design["front_camber_deg"])
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


CASE = _case(
    "double_rotor_fan",
    "Counter-rotating fan",
    double_rotor_fan,
    (
        variable("front_camber_deg", 4.0, 16.0, 9.0),
        variable("rear_camber_deg", 4.0, 16.0, 9.0),
        variable("speed_ratio", 0.7, 1.2, 1.0),
    ),
    "efficiency",
    (at_least("pressure_rise_pa"),),
    maximize=True,
    positive=_PUMPING,
    simplification=_EQUAL,
)
