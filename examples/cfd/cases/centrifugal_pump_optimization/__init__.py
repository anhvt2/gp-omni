"""Pump impeller and casing shape."""

from foam.case import at_least, variable

from ..radial import _CASING, _PUMPING, _WATER_PUMP, WATER, _case, _impeller


def centrifugal_pump_optimization(case, design, level):
    return _impeller(
        case, level, **_WATER_PUMP, width=design["outlet_width_m"],
        outlet_angle=design["outlet_angle_deg"], inlet_offset=design["inlet_offset_deg"],
        **WATER,
    )  # fmt: skip


CASE = _case(
    "centrifugal_pump_optimization",
    "Pump impeller and casing shape",
    centrifugal_pump_optimization,
    (
        variable("outlet_angle_deg", 40.0, 65.0, 55.0),
        variable("outlet_width_m", 0.016, 0.028, 0.022),
        variable("inlet_offset_deg", -4.0, 6.0, 2.0),
    ),
    "efficiency",
    (at_least("head_m"),),
    maximize=True,
    positive=_PUMPING,
    simplification=_CASING,
)
