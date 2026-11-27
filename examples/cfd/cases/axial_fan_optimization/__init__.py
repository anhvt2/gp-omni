"""Axial fan with a free blade count."""

from foam.case import at_least, variable

from ..axial import _PUMPING, _case, _fan


def axial_fan_optimization(case, design, level):
    return _fan(
        case,
        level,
        design["camber_deg"],
        design["inlet_offset_deg"],
        0.25,
        round(design["blade_count"]),
    )


CASE = _case(
    "axial_fan_optimization",
    "Axial fan with a free blade count",
    axial_fan_optimization,
    (
        variable("inlet_offset_deg", -3.0, 6.0, 2.0),
        variable("camber_deg", 4.0, 18.0, 10.0),
        variable("blade_count", 6.0, 12.0, 9.0),
    ),
    "efficiency",
    (at_least("pressure_rise_pa"),),
    maximize=True,
    positive=_PUMPING,
    simplification="shrouded blades without tip clearance",
)
