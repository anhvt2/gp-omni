"""Radial and mixed-flow turbomachinery.

Impellers have two-dimensional blades and a rectangular spiral casing meshed together with
the passages. Turbines reverse the meridional channel so that the flow runs inward.
"""

import importlib.util
import shutil
from functools import cache
from math import atan, degrees, pi, radians, tan
from pathlib import Path

import numpy as np
from foam import measure as post
from foam.case import at_least, cfd_case, run_plan, variable
from foam.passage import blade_row, machine, radial_channel, volute

from .turbo import build_machine, duty, relative_angle

PUMP_SCRIPT = Path(__file__).resolve().parents[1] / "centrifugal_pump.py"


@cache
def _pump_driver():
    """The stand-alone pump example, whose bundled mesh and controlDict this case reuses."""
    spec = importlib.util.spec_from_file_location("centrifugal_pump", PUMP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def centrifugal_pump(case, design, level):
    pump = _pump_driver()
    pump.verify_case()
    for folder in ("0", "constant", "system"):
        shutil.copytree(pump.BUNDLED_CASE / folder, case / folder, dirs_exist_ok=True)
    (case / "system/controlDict").write_text(pump.control_dict(level.iterations))
    speed = pump.REFERENCE_SPEED_RAD_S * design["speed_ratio"]
    scale = design["scale"]
    mesh = [
        ["foamDictionary", "constant/MRFProperties", "-entry", "MRF1/omega",
         "-set", f"{-speed:.10g}"],
        ["transformPoints", "-scale", f"({scale} {scale} {scale})"],
    ]  # fmt: skip

    def measure(directory: Path) -> dict[str, float]:
        inlet_p, outlet_p, inlet_q, outlet_q = (
            post.last_surface(directory, name)
            for name in ("inletPressure", "outletPressure", "inletFlow", "outletFlow")
        )
        moment = post.last_moment(directory, "impellerForces")
        return {
            "flow_imbalance": abs(inlet_q + outlet_q) / max(abs(inlet_q), abs(outlet_q), 1e-12),
            "flow_m3_s": outlet_q,
            "pressure_rise_pa": pump.DENSITY_KG_M3 * (outlet_p - inlet_p),
            "shaft_power_w": float(speed * np.asarray(moment)[0]),
        }

    return run_plan(mesh=mesh, solve=[["simpleFoam"]], measure=measure)


def _impeller(
    case, level, *, eye_hub, eye_tip, outer, width, depth, blades, omega, speed,
    outlet_angle, inlet_offset=0.0, casing_height=None, casing_growth=None, offset=0.0,
    sliding=False, **point_kwargs,
):  # fmt: skip
    """A backswept impeller discharging into a spiral casing."""
    extension = 0.12 * outer
    hub, shroud = radial_channel(
        eye_hub, eye_tip, outer, width, depth, inlet=0.4 * outer, outlet=extension
    )
    eye_radius = 0.5 * (eye_hub + eye_tip) + 0.1 * (outer - eye_tip)
    inlet_angle = relative_angle(omega, eye_radius, speed) + inlet_offset
    row = blade_row(
        "rotor", 0.4, 0.86, (inlet_angle,) * 2, (-outlet_angle,) * 2, thickness=0.08,
        omega=omega,
    )  # fmt: skip
    casing = volute(
        height=casing_height or 0.1 * outer,
        growth=casing_growth or 0.35 * outer,
        offset=offset,
    )
    wheel = machine(
        hub, shroud, blades=blades, rows=(row,), full_wheel=True, spiral=casing,
        sliding=sliding,
    )  # fmt: skip
    flow = speed * pi * (eye_tip**2 - eye_hub**2) * point_kwargs.pop("density_inlet", 1.0)
    point = duty("pump", speed, 0.3 * outer, flow=flow, **point_kwargs)
    return build_machine(case, wheel, point, level)


WATER = {"density": 1000.0, "viscosity": 1e-6}
_WATER_PUMP = {
    "eye_hub": 0.03, "eye_tip": 0.07, "outer": 0.14, "depth": 0.05, "blades": 7,
    "omega": 105.0, "speed": 3.0,
}  # fmt: skip


def centrifugal_pump_optimization(case, design, level):
    return _impeller(
        case, level, **_WATER_PUMP, width=design["outlet_width_m"],
        outlet_angle=design["outlet_angle_deg"], inlet_offset=design["inlet_offset_deg"],
        **WATER,
    )  # fmt: skip


def centrifugal_pump_partial_overlap(case, design, level):
    base = _WATER_PUMP | {"omega": 105.0 * design["speed_ratio"]}
    return _impeller(
        case, level, **base, width=0.022, outlet_angle=55.0,
        offset=design["outlet_offset"], sliding=True, transient=True, time_step=3e-4,
        partial_overlap=True, **WATER,
    )  # fmt: skip


def fda_pump(case, design, level):
    return _impeller(
        case, level, eye_hub=0.004, eye_tip=0.011, outer=0.026, width=0.004, depth=0.009,
        blades=5, omega=260.0 * design["speed_ratio"], speed=0.35,
        outlet_angle=design["outlet_angle_deg"], casing_height=design["casing_gap_m"],
        casing_growth=0.008, density=1035.0, viscosity=3.4e-6, wall_shear=True,
    )  # fmt: skip


def centrifugal_fan(case, design, level):
    return _impeller(
        case, level, eye_hub=0.08, eye_tip=0.16, outer=0.26, width=design["outlet_width_m"],
        depth=0.12, blades=round(design["blade_count"]), omega=100.0, speed=9.0,
        outlet_angle=design["outlet_angle_deg"],
    )  # fmt: skip


def radial_compressor(case, design, level):
    return _impeller(
        case, level, eye_hub=0.02, eye_tip=0.05, outer=0.09,
        width=design["outlet_width_m"], depth=0.035, blades=11, omega=2200.0, speed=35.0,
        outlet_angle=design["backsweep_deg"], compressible=True, density_inlet=1.17,
    )  # fmt: skip


def hecc_stage(case, design, level):
    outer, omega, speed = 0.09, 2200.0 * design["speed_ratio"], 70.0
    hub, shroud = radial_channel(
        0.02, 0.05, outer, 0.009, 0.035, inlet=0.03, outlet=0.07
    )
    inlet_angle = relative_angle(omega, 0.039, speed)
    impeller = blade_row(
        "rotor", 0.2, 0.55, (inlet_angle,) * 2, (-35.0,) * 2, thickness=0.06, omega=omega
    )
    angle = design["diffuser_angle_deg"]
    diffuser = blade_row(
        "diffuser", 0.66, 0.88, (angle,) * 2, (angle - 10,) * 2, thickness=0.05
    )
    wheel = machine(hub, shroud, blades=15, rows=(impeller, diffuser))
    flow = 1.17 * speed * pi * (0.05**2 - 0.02**2)
    point = duty("pump", speed, 0.03, flow=flow, compressible=True)
    return build_machine(case, wheel, point, level)


def _inflow_turbine(
    case, level, *, eye_hub, eye_tip, runner, width, depth, blades, omega, flow, vane_angle,
    exit_offset, density_inlet=1.0, **point_kwargs,
):  # fmt: skip
    """Spiral casing, nozzle vanes and an inward-flow runner discharging axially."""
    ring = 0.4 * runner
    hub, shroud = radial_channel(
        eye_hub, eye_tip, runner, width, depth, inlet=0.45 * runner, outlet=ring
    )
    hub, shroud = hub[::-1], shroud[::-1]  # the flow now runs inward
    volume_flow = flow / density_inlet
    inflow = volume_flow / (2 * pi * runner * width)
    exit_speed = volume_flow / (pi * (eye_tip**2 - eye_hub**2))
    swirl = inflow * tan(radians(vane_angle))
    runner_in = degrees(atan((swirl - omega * runner) / inflow))
    runner_out = relative_angle(omega, 0.5 * (eye_hub + eye_tip), exit_speed) + exit_offset
    vanes = blade_row(
        "vane", 0.06, 0.24, (0.35 * vane_angle,) * 2, (vane_angle,) * 2, thickness=0.06
    )
    rotor = blade_row(
        "rotor", 0.36, 0.75, (runner_in,) * 2, (runner_out,) * 2, thickness=0.06, omega=omega
    )
    wheel = machine(
        hub, shroud, blades=blades, rows=(vanes, rotor), full_wheel=True,
        spiral=volute(0.15 * runner, 0.4 * runner, at="start"),
    )  # fmt: skip
    point = duty("turbine", inflow, 0.2 * runner, flow=flow, **point_kwargs)
    return build_machine(case, wheel, point, level)


def francis_turbine(case, design, level):
    return _inflow_turbine(
        case, level, eye_hub=0.06, eye_tip=0.15, runner=0.22, width=0.06, depth=0.13,
        blades=11, omega=36.0 * design["speed_ratio"], flow=0.4,
        vane_angle=design["vane_angle_deg"], exit_offset=design["exit_offset_deg"], **WATER,
    )  # fmt: skip


def radial_turbine(case, design, level):
    density = 101325.0 / (287.0 * 600.0)
    return _inflow_turbine(
        case, level, eye_hub=0.012, eye_tip=0.04, runner=0.06, width=0.008, depth=0.03,
        blades=11, omega=2200.0, flow=0.08, vane_angle=design["vane_angle_deg"],
        exit_offset=design["exit_offset_deg"], density_inlet=density, compressible=True,
        inlet_temperature=600.0,
    )  # fmt: skip


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    mesher = options.pop("mesher", "structured")
    return cfd_case(
        name, "radial_turbo", title, mesher, solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip


_CASING = "two-dimensional blades, rectangular spiral casing with a radial slot over half a turn"
_PUMPING = ("pressure_rise_pa", "shaft_power_w")
_TURBINE = ("output_power_w",)

CASES = [
    _case(
        "centrifugal_pump", "Volute water pump on the bundled reference mesh", centrifugal_pump,
        (variable("scale", 0.9, 1.1, 1.0), variable("speed_ratio", 0.85, 1.15, 1.0)),
        "shaft_power_w", (at_least("pressure_rise_pa"),), mesher="bundled", positive=_PUMPING,
    ),
    _case(
        "centrifugal_pump_optimization", "Pump impeller and casing shape",
        centrifugal_pump_optimization,
        (
            variable("outlet_angle_deg", 40.0, 65.0, 55.0),
            variable("outlet_width_m", 0.016, 0.028, 0.022),
            variable("inlet_offset_deg", -4.0, 6.0, 2.0),
        ),
        "efficiency", (at_least("head_m"),), maximize=True, positive=_PUMPING,
        simplification=_CASING,
    ),
    _case(
        "centrifugal_pump_partial_overlap", "Sliding pump with an offset casing",
        centrifugal_pump_partial_overlap,
        (variable("outlet_offset", 0.1, 0.4, 0.25), variable("speed_ratio", 0.85, 1.15, 1.0)),
        "shaft_power_w", (at_least("pressure_rise_pa", 0.3),), solver="pimpleFoam",
        simplification=_CASING + ", short transient",
    ),
    _case(
        "fda_pump", "Small centrifugal blood pump", fda_pump,
        (
            variable("speed_ratio", 0.8, 1.2, 1.0),
            variable("outlet_angle_deg", 20.0, 55.0, 40.0),
            variable("casing_gap_m", 0.0015, 0.004, 0.0025),
        ),
        "wall_shear_pa", (at_least("pressure_rise_pa", 0.15),), positive=_PUMPING,
        simplification=_CASING + ", own geometry",
    ),
    _case(
        "centrifugal_fan", "Backswept centrifugal fan", centrifugal_fan,
        (
            variable("outlet_angle_deg", 30.0, 62.0, 48.0),
            variable("blade_count", 8.0, 12.0, 10.0),
            variable("outlet_width_m", 0.06, 0.1, 0.08),
        ),
        "efficiency", (at_least("pressure_rise_pa"),), maximize=True, positive=_PUMPING,
        simplification=_CASING,
    ),
    _case(
        "francis_turbine", "Francis turbine with casing and guide vanes", francis_turbine,
        (
            variable("vane_angle_deg", 48.0, 66.0, 58.0),
            variable("exit_offset_deg", -8.0, 14.0, 2.0),
            variable("speed_ratio", 0.8, 1.2, 1.0),
        ),
        "efficiency", (at_least("output_power_w"),), maximize=True, positive=_TURBINE,
        simplification=_CASING + " used as the inlet, no draft tube",
    ),
    _case(
        "radial_turbine", "Hot-gas radial inflow turbine", radial_turbine,
        (
            variable("vane_angle_deg", 48.0, 66.0, 58.0),
            variable("exit_offset_deg", -8.0, 14.0, 2.0),
        ),
        "efficiency", (at_least("output_power_w"),), maximize=True, positive=_TURBINE,
        solver="rhoSimpleFoam", simplification=_CASING + " used as the inlet",
    ),
    _case(
        "radial_compressor", "Centrifugal compressor with casing", radial_compressor,
        (
            variable("backsweep_deg", 20.0, 50.0, 35.0),
            variable("outlet_width_m", 0.006, 0.011, 0.008),
        ),
        "efficiency", (at_least("pressure_ratio", 0.01),), maximize=True,
        positive=("shaft_power_w",), solver="rhoSimpleFoam", simplification=_CASING,
    ),
    _case(
        "hecc_stage", "Compressor stage with a vaned diffuser", hecc_stage,
        (
            variable("speed_ratio", 0.8, 1.1, 1.0),
            variable("diffuser_angle_deg", 50.0, 66.0, 60.0),
        ),
        "efficiency", (at_least("pressure_ratio", 0.01),), maximize=True,
        positive=("shaft_power_w",), solver="rhoSimpleFoam",
        simplification="representative stage with equal impeller and diffuser blade counts",
    ),
]
