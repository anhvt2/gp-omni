"""Three-blade propeller sizing."""

from foam.case import at_least, variable

from ..axial import _OPEN, _PUSHING, _case, _open_rotor


def propeller_optimization(case, design, level):
    radius = design["diameter_m"] / 2
    omega = 500.0
    return _open_rotor(
        case, level, "propeller", radius, 0.12 * radius, 3, 500.0 * 0.12 / 2.5, omega,
        design["pitch_deg"], design["chord_fraction"],
    )  # fmt: skip


CASE = _case(
    "propeller_optimization",
    "Three-blade propeller sizing",
    propeller_optimization,
    (
        variable("pitch_deg", 2.0, 10.0, 5.0),
        variable("chord_fraction", 0.08, 0.16, 0.12),
        variable("diameter_m", 0.2, 0.28, 0.24),
    ),
    "shaft_power_w",
    (at_least("thrust_n"),),
    positive=_PUSHING,
    simplification=_OPEN + ", fixed axial inflow in place of static hover",
)
