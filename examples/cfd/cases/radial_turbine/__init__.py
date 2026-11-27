"""Hot-gas radial inflow turbine."""

from foam.case import at_least, variable

from ..radial import _CASING, _TURBINE, _case, _inflow_turbine


def radial_turbine(case, design, level):
    density = 101325.0 / (287.0 * 600.0)
    return _inflow_turbine(
        case, level, eye_hub=0.012, eye_tip=0.04, runner=0.06, width=0.008, depth=0.03,
        blades=11, omega=2200.0, flow=0.08, vane_angle=design["vane_angle_deg"],
        exit_offset=design["exit_offset_deg"], density_inlet=density, compressible=True,
        inlet_temperature=600.0,
    )  # fmt: skip


CASE = _case(
    "radial_turbine",
    "Hot-gas radial inflow turbine",
    radial_turbine,
    (
        variable("vane_angle_deg", 48.0, 66.0, 58.0),
        variable("exit_offset_deg", -8.0, 14.0, 2.0),
    ),
    "efficiency",
    (at_least("output_power_w"),),
    maximize=True,
    positive=_TURBINE,
    solver="rhoSimpleFoam",
    simplification=_CASING + " used as the inlet",
)
