"""Kaplan runner behind guide vanes."""

from foam.case import at_least, variable

from ..axial import _EQUAL, _case, _turbine


def kaplan_turbine(case, design, level):
    return _turbine(
        case, level, 0.1, 0.22, 35.0 * design["speed_ratio"], 5.0, design["guide_angle_deg"],
        design["runner_offset_deg"], 11, 0.35, density=1000.0, viscosity=1e-6,
    )  # fmt: skip


CASE = _case(
    "kaplan_turbine",
    "Kaplan runner behind guide vanes",
    kaplan_turbine,
    (
        variable("guide_angle_deg", 40.0, 62.0, 52.0),
        variable("runner_offset_deg", -6.0, 14.0, 4.0),
        variable("speed_ratio", 0.75, 1.2, 1.0),
    ),
    "efficiency",
    (at_least("output_power_w"),),
    maximize=True,
    positive=("output_power_w", "pressure_drop_pa"),
    simplification=_EQUAL + ", axial guide vanes, no spiral or draft tube",
)
