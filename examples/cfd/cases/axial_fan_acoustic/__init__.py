"""Sliding-mesh axial fan with pressure probes."""

from foam.case import at_least, variable

from ..axial import _case, _fan


def axial_fan_acoustic(case, design, level):
    probes = ((0.19, 0.01, 0.03), (0.01, 0.19, 0.27))
    kwargs = {"transient": True, "time_step": 2.3e-4, "probes": probes}
    return _fan(
        case, level, 10.0, 2.0, 0.25, round(design["blade_count"]), kwargs, sliding=True,
        speed_ratio=design["speed_ratio"],
    )  # fmt: skip


CASE = _case(
    "axial_fan_acoustic",
    "Sliding-mesh axial fan with pressure probes",
    axial_fan_acoustic,
    (variable("speed_ratio", 0.8, 1.15, 1.0), variable("blade_count", 5.0, 9.0, 7.0)),
    "pressure_fluctuation_pa",
    (at_least("pressure_rise_pa", 0.3),),
    solver="pimpleFoam",
    simplification="probe pressures only, no far-field acoustics",
)
