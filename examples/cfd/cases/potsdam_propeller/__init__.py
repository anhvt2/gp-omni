"""Five-blade propeller advance sweep."""

from foam.case import at_least, variable

from ..axial import _OPEN, _PUSHING, _case, _open_rotor


def potsdam_propeller(case, design, level):
    return _open_rotor(
        case, level, "propeller", 0.125, 0.0375, 5, design["inflow_m_s"], 94.25,
        design["pitch_deg"], 0.14, 1000.0, 1e-6,
    )  # fmt: skip


CASE = _case(
    "potsdam_propeller",
    "Five-blade propeller advance sweep",
    potsdam_propeller,
    (variable("pitch_deg", 2.0, 10.0, 5.0), variable("inflow_m_s", 4.0, 6.0, 4.7)),
    "efficiency",
    (at_least("thrust_n"),),
    maximize=True,
    positive=_PUSHING,
    simplification=_OPEN + ", own blade shape, no cavitation",
)
