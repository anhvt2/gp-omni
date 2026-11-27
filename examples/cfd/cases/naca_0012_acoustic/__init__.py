"""Unsteady pressure near the NACA 0012 trailing edge."""

from foam.case import at_least, variable

from ..external_structured import _airfoil, _case


def naca_0012_acoustic(case, design, level):
    return _airfoil(case, level, design["angle_deg"], design["speed_m_s"], time_step=2e-5)


CASE = _case(
    "naca_0012_acoustic",
    "Unsteady pressure near the NACA 0012 trailing edge",
    naca_0012_acoustic,
    (variable("angle_deg", 0.0, 10.0, 5.0), variable("speed_m_s", 40.0, 70.0, 55.0)),
    "pressure_fluctuation_pa",
    (at_least("lift_coefficient", 0.4),),
    solver="pimpleFoam",
    simplification="two-dimensional short transient, probes only",
)
