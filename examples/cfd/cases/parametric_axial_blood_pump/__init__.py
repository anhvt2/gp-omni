"""Miniature axial blood pump."""

from foam.case import at_least, variable

from ..axial import _EQUAL, _PUMPING, _case, _stage_pump


def parametric_axial_blood_pump(case, design, level):
    tip = 0.008
    return _stage_pump(
        case, level, design["hub_ratio"] * tip, tip, design["speed_rad_s"], 1.4,
        design["pitch_deg"], 12.0, 0.0, 5, 0.03, density=1055.0, viscosity=3.3e-6,
        wall_shear=True,
    )  # fmt: skip


CASE = _case(
    "parametric_axial_blood_pump",
    "Miniature axial blood pump",
    parametric_axial_blood_pump,
    (
        variable("pitch_deg", 0.0, 8.0, 4.0),
        variable("hub_ratio", 0.4, 0.62, 0.5),
        variable("speed_rad_s", 500.0, 800.0, 650.0),
    ),
    "wall_shear_pa",
    (at_least("pressure_rise_pa", 0.15),),
    positive=_PUMPING,
    simplification=_EQUAL + ", wall shear in place of a blood damage model",
)
