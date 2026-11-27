"""Small centrifugal blood pump."""

from foam.case import at_least, variable

from ..radial import _CASING, _PUMPING, _case, _impeller


def fda_pump(case, design, level):
    return _impeller(
        case, level, eye_hub=0.004, eye_tip=0.011, outer=0.026, width=0.004, depth=0.009,
        blades=5, omega=260.0 * design["speed_ratio"], speed=0.35,
        outlet_angle=design["outlet_angle_deg"], casing_height=design["casing_gap_m"],
        casing_growth=0.008, density=1035.0, viscosity=3.4e-6, wall_shear=True,
    )  # fmt: skip


CASE = _case(
    "fda_pump",
    "Small centrifugal blood pump",
    fda_pump,
    (
        variable("speed_ratio", 0.8, 1.2, 1.0),
        variable("outlet_angle_deg", 20.0, 55.0, 40.0),
        variable("casing_gap_m", 0.0015, 0.004, 0.0025),
    ),
    "wall_shear_pa",
    (at_least("pressure_rise_pa", 0.15),),
    positive=_PUMPING,
    simplification=_CASING + ", own geometry",
)
