"""Centrifugal compressor with casing."""

from foam.case import at_least, variable

from ..radial import _CASING, _case, _impeller


def radial_compressor(case, design, level):
    return _impeller(
        case, level, eye_hub=0.02, eye_tip=0.05, outer=0.09,
        width=design["outlet_width_m"], depth=0.035, blades=11, omega=2200.0, speed=35.0,
        outlet_angle=design["backsweep_deg"], compressible=True, density_inlet=1.17,
    )  # fmt: skip


CASE = _case(
    "radial_compressor",
    "Centrifugal compressor with casing",
    radial_compressor,
    (
        variable("backsweep_deg", 20.0, 50.0, 35.0),
        variable("outlet_width_m", 0.006, 0.011, 0.008),
    ),
    "efficiency",
    (at_least("pressure_ratio", 0.01),),
    maximize=True,
    positive=("shaft_power_w",),
    solver="rhoSimpleFoam",
    simplification=_CASING,
)
