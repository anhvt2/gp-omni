"""Ducted axial fan, one blade passage."""

from foam.case import at_least, variable

from ..axial import _PUMPING, _case, _fan


def axial_fan(case, design, level):
    return _fan(
        case, level, design["camber_deg"], design["inlet_offset_deg"], design["chord_fraction"], 9
    )


CASE = _case(
    "axial_fan",
    "Ducted axial fan, one blade passage",
    axial_fan,
    (
        variable("inlet_offset_deg", -3.0, 6.0, 2.0),
        variable("camber_deg", 4.0, 18.0, 10.0),
        variable("chord_fraction", 0.18, 0.32, 0.25),
    ),
    "efficiency",
    (at_least("pressure_rise_pa"),),
    maximize=True,
    positive=_PUMPING,
    simplification="shrouded blades without tip clearance",
)
