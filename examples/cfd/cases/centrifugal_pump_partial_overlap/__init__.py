"""Sliding pump with an offset casing."""

from foam.case import at_least, variable

from ..radial import _CASING, _WATER_PUMP, WATER, _case, _impeller


def centrifugal_pump_partial_overlap(case, design, level):
    base = _WATER_PUMP | {"omega": 105.0 * design["speed_ratio"]}
    return _impeller(
        case, level, **base, width=0.022, outlet_angle=55.0,
        offset=design["outlet_offset"], sliding=True, transient=True, time_step=3e-4,
        partial_overlap=True, **WATER,
    )  # fmt: skip


CASE = _case(
    "centrifugal_pump_partial_overlap",
    "Sliding pump with an offset casing",
    centrifugal_pump_partial_overlap,
    (variable("outlet_offset", 0.1, 0.4, 0.25), variable("speed_ratio", 0.85, 1.15, 1.0)),
    "shaft_power_w",
    (at_least("pressure_rise_pa", 0.3),),
    solver="pimpleFoam",
    simplification=_CASING + ", short transient",
)
