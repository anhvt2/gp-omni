"""Volute water pump on the bundled reference mesh."""

import importlib.util
import shutil
from functools import cache
from pathlib import Path

import numpy as np
from foam import measure as post
from foam.case import at_least, run_plan, variable

from ..radial import _PUMPING, _case

PUMP_SCRIPT = Path(__file__).resolve().parents[2] / "centrifugal_pump.py"


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


CASE = _case(
    "centrifugal_pump",
    "Volute water pump on the bundled reference mesh",
    centrifugal_pump,
    (variable("scale", 0.9, 1.1, 1.0), variable("speed_ratio", 0.85, 1.15, 1.0)),
    "shaft_power_w",
    (at_least("pressure_rise_pa"),),
    mesher="bundled",
    positive=_PUMPING,
)
