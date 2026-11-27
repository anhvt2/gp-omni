"""Intake manifold with exhaust gas recirculation."""

import numpy as np
from foam import measure as post
from foam.boundaries import flow_inlet, pressure_outlet, wall
from foam.cad import fluid_surfaces, fuse, name_caps, tube
from foam.case import at_most, run_plan, variable

from ..internal_cad import AIR_DENSITY, _case, _snappy_fluid, _steady, _total

RUNNERS = (0.05, 0.13, 0.21, 0.29)


def intake_manifold_egr(case, design, level):
    """Fresh air and recirculated gas mixing in a plenum feeding four runners."""
    plenum, runner = design["plenum_radius_m"], design["runner_radius_m"]
    port = design["egr_position_m"]

    def build(occ):
        parts = [
            tube(occ, (0.0, 0, 0), (0.34, 0, 0), plenum),
            tube(occ, (-0.14, 0, 0), (0.005, 0, 0), 0.025),
            tube(occ, (port, 0.07, 0), (port, 0, 0), 0.012),
        ]
        parts += [tube(occ, (x, 0, 0), (x, 0, -0.18), runner) for x in RUNNERS]
        return fuse(occ, parts)

    caps = {"inlet": (-0.14, 0, 0), "egr": (port, 0.07, 0)}
    caps |= {f"runner_{n}": (x, 0, -0.18) for n, x in enumerate(RUNNERS, 1)}
    surfaces, _ = fluid_surfaces(build, name_caps(caps), 0.008)
    air, gas = 0.036, 0.004
    outlets = [f"runner_{n}" for n in range(1, 5)]
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {
            "inlet": flow_inlet(air, egr=0.0), "egr": flow_inlet(gas, egr=1.0),
            **{n: pressure_outlet() for n in outlets}, "walls": wall(),
        },
        0.009, (0.1713, 0.0029, 0.0031), (1, 1),
    )  # fmt: skip
    patches_all = ("inlet", "egr", *outlets)
    functions = post.flux(*patches_all) | post.total_pressure("inlet", *outlets)
    functions |= post.scalar_transport("egr")
    for name in outlets:
        functions |= post.surface_value(f"egr_{name}", name, "weightedAverage", "egr", weight="phi")
    _steady(case, patches, 20.0, 0.05, level, functions, scalars=("egr",))

    def measure(directory):
        share = np.array([post.last_surface(directory, f"egr_{n}") for n in outlets])
        flows = np.array([abs(post.last_flux(directory, n)) for n in outlets])
        outlet_total = np.average([_total(directory, n) for n in outlets], weights=flows)
        return {
            "flow_imbalance": post.imbalance(directory, *patches_all),
            "mixture_imbalance": float(np.ptp(share) / max(abs(share.mean()), 1e-9)),
            "flow_imbalance_between_runners": float(np.ptp(flows) / flows.mean()),
            "pressure_loss_pa": AIR_DENSITY * float(_total(directory, "inlet") - outlet_total),
        }

    return run_plan(mesh=commands, solve=[["simpleFoam"]], measure=measure)


CASE = _case(
    "intake_manifold_egr",
    "Intake manifold with exhaust gas recirculation",
    intake_manifold_egr,
    (
        variable("plenum_radius_m", 0.032, 0.05, 0.04),
        variable("runner_radius_m", 0.015, 0.022, 0.018),
        variable("egr_position_m", -0.08, -0.025, -0.05),
    ),
    "mixture_imbalance",
    (at_most("pressure_loss_pa", 0.2),),
    positive=("pressure_loss_pa",),
    simplification="steady flow, passive scalar mixing",
)
