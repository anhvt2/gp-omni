"""Axial water pump with stator."""

from foam.case import at_least, variable

from ..axial import _EQUAL, _PUMPING, _case, _stage_pump


def axial_pump(case, design, level):
    return _stage_pump(
        case, level, 0.06, 0.12, 120.0, 6.5, design["incidence_deg"], design["camber_deg"],
        design["stator_offset_deg"], 7, 0.2, density=1000.0, viscosity=1e-6,
    )  # fmt: skip


CASE = _case(
    "axial_pump",
    "Axial water pump with stator",
    axial_pump,
    (
        variable("incidence_deg", -2.0, 6.0, 2.0),
        variable("camber_deg", 5.0, 16.0, 10.0),
        variable("stator_offset_deg", -6.0, 6.0, 0.0),
    ),
    "efficiency",
    (at_least("pressure_rise_pa"),),
    maximize=True,
    positive=_PUMPING,
    simplification=_EQUAL,
)
