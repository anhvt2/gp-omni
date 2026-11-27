"""NACA 0012 airfoil in two dimensions."""

from foam.case import at_least, variable

from ..external_structured import _airfoil, _case


def naca_0012(case, design, level):
    return _airfoil(case, level, design["angle_deg"], 51.0)


CASE = _case(
    "naca_0012",
    "NACA 0012 airfoil in two dimensions",
    naca_0012,
    (variable("angle_deg", 1.0, 12.0, 5.0),),
    "lift_to_drag",
    (at_least("lift_coefficient", 0.2),),
    maximize=True,
    positive=("drag_coefficient", "lift_coefficient"),
)
