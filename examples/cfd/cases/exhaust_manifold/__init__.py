"""Four-into-one exhaust manifold."""

import numpy as np
from foam import measure as post
from foam.boundaries import flow_inlet, pressure_outlet, wall
from foam.cad import fluid_surfaces, fuse, name_caps, tube
from foam.case import at_most, run_plan, variable
from foam.physics import flow, write_flow_case

from ..internal_cad import _case, _snappy_fluid


def exhaust_manifold(case, design, level):
    """Hot gas from four ports gathering into one collector, a short compressible transient."""
    runner, collector = design["runner_radius_m"], design["collector_radius_m"]
    outlet_x = design["collector_position_m"]
    ports = (0.0, 0.09, 0.18, 0.27)

    def build(occ):
        parts = [tube(occ, (0.0, 0, 0), (0.27, 0, 0), collector)]
        parts += [tube(occ, (x, 0.12, 0), (x, 0, 0), runner) for x in ports]
        parts.append(tube(occ, (outlet_x, 0, 0), (outlet_x, 0, -0.15), collector))
        return fuse(occ, parts)

    inlets = [f"port_{n}" for n in range(1, 5)]
    caps = {n: (x, 0.12, 0) for n, x in zip(inlets, ports)}
    caps["outlet"] = (outlet_x, 0, -0.15)
    surfaces, volume = fluid_surfaces(build, name_caps(caps), 0.008)
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {
            **{n: flow_inlet(0.008, temperature=800.0) for n in inlets},
            "outlet": pressure_outlet(101325.0), "walls": wall(),
        },
        0.008, (0.0451, 0.0019, 0.0023), (1, 1),
    )  # fmt: skip
    functions = post.flux(*inlets, "outlet")
    functions |= post.compressible_total_pressure(*inlets, "outlet")
    write_flow_case(
        case,
        patches,
        flow(
            "rhoPimpleFoam", velocity=50.0, length=2 * runner, time_step=1e-4,
            time_steps=level.time_steps, temperature=800.0, functions=functions,
        ),
    )  # fmt: skip

    def measure(directory):
        # pressure waves still travel through the manifold, so average over time
        inlet = np.mean([post.mean_surface(directory, f"total_p_{n}") for n in inlets])
        outlet = post.mean_surface(directory, "total_p_outlet")
        return {
            "flow_mismatch": post.imbalance(directory, *inlets, "outlet"),
            "pressure_loss_pa": float(inlet - outlet),
            "volume_m3": volume,
        }

    return run_plan(mesh=commands, solve=[["rhoPimpleFoam"]], measure=measure)


CASE = _case(
    "exhaust_manifold",
    "Four-into-one exhaust manifold",
    exhaust_manifold,
    (
        variable("runner_radius_m", 0.014, 0.022, 0.018),
        variable("collector_radius_m", 0.022, 0.034, 0.028),
        variable("collector_position_m", 0.06, 0.21, 0.135),
    ),
    "pressure_loss_pa",
    (at_most("volume_m3"),),
    solver="rhoPimpleFoam",
    positive=("pressure_loss_pa",),
    simplification="steady port flows, short transient",
)
