"""Axial turbine stage."""

from foam.case import at_least, variable

from ..axial import _EQUAL, _case, _turbine


def axial_turbine(case, design, level):
    return _turbine(
        case, level, 0.2, 0.3, 120.0 * design["speed_ratio"], 20.0, design["vane_angle_deg"],
        design["exit_offset_deg"], 26, 0.25,
    )  # fmt: skip


CASE = _case(
    "axial_turbine",
    "Axial turbine stage",
    axial_turbine,
    (
        variable("vane_angle_deg", 50.0, 68.0, 60.0),
        variable("exit_offset_deg", -6.0, 14.0, 4.0),
        variable("speed_ratio", 0.75, 1.2, 1.0),
    ),
    "efficiency",
    (at_least("output_power_w"),),
    maximize=True,
    positive=("output_power_w", "pressure_drop_pa"),
    simplification=_EQUAL + ", no spiral casing",
)
