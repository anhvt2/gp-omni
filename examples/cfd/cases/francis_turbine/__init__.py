"""Francis turbine with casing and guide vanes."""

from foam.case import at_least, variable

from ..radial import _CASING, _TURBINE, WATER, _case, _inflow_turbine


def francis_turbine(case, design, level):
    return _inflow_turbine(
        case, level, eye_hub=0.06, eye_tip=0.15, runner=0.22, width=0.06, depth=0.13,
        blades=11, omega=36.0 * design["speed_ratio"], flow=0.4,
        vane_angle=design["vane_angle_deg"], exit_offset=design["exit_offset_deg"], **WATER,
    )  # fmt: skip


CASE = _case(
    "francis_turbine",
    "Francis turbine with casing and guide vanes",
    francis_turbine,
    (
        variable("vane_angle_deg", 48.0, 66.0, 58.0),
        variable("exit_offset_deg", -8.0, 14.0, 2.0),
        variable("speed_ratio", 0.8, 1.2, 1.0),
    ),
    "efficiency",
    (at_least("output_power_w"),),
    maximize=True,
    positive=_TURBINE,
    simplification=_CASING + " used as the inlet, no draft tube",
)
