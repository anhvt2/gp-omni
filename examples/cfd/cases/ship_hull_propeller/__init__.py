"""Marine propeller in open water."""

from foam.case import at_least, variable

from ..axial import _OPEN, _PUSHING, _case, _open_rotor


def ship_hull_propeller(case, design, level):
    radius, omega = 0.125, 50.0 * design["speed_ratio"]
    return _open_rotor(
        case, level, "propeller", radius, 0.025, 4, 2.5, omega, design["pitch_deg"],
        design["chord_fraction"], 1000.0, 1e-6,
    )  # fmt: skip


CASE = _case(
    "ship_hull_propeller",
    "Marine propeller in open water",
    ship_hull_propeller,
    (
        variable("pitch_deg", 2.0, 10.0, 5.0),
        variable("chord_fraction", 0.1, 0.18, 0.14),
        variable("speed_ratio", 0.85, 1.1, 1.0),
    ),
    "efficiency",
    (at_least("thrust_n"),),
    maximize=True,
    positive=_PUSHING,
    simplification=_OPEN + ", no hull",
)
