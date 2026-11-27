"""Backswept centrifugal fan."""

from foam.case import at_least, variable

from ..radial import _CASING, _PUMPING, _case, _impeller


def centrifugal_fan(case, design, level):
    return _impeller(
        case, level, eye_hub=0.08, eye_tip=0.16, outer=0.26, width=design["outlet_width_m"],
        depth=0.12, blades=round(design["blade_count"]), omega=100.0, speed=9.0,
        outlet_angle=design["outlet_angle_deg"],
    )  # fmt: skip


CASE = _case(
    "centrifugal_fan",
    "Backswept centrifugal fan",
    centrifugal_fan,
    (
        variable("outlet_angle_deg", 30.0, 62.0, 48.0),
        variable("blade_count", 8.0, 12.0, 10.0),
        variable("outlet_width_m", 0.06, 0.1, 0.08),
    ),
    "efficiency",
    (at_least("pressure_rise_pa"),),
    maximize=True,
    positive=_PUMPING,
    simplification=_CASING,
)
