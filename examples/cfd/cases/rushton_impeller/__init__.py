"""Rushton turbine in a stirred vessel."""

from math import pi

import numpy as np
from foam import measure as post
from foam.boundaries import rotating_wall, slip, wall
from foam.case import at_least, run_plan, variable
from foam.hex_mesh import hex_mesh
from foam.physics import flow, rotation, write_flow_case

from ..internal import WATER, _case, _finish, _polar_grid


def rushton_impeller(case, design, level):
    """An unbaffled 800 mm vessel with a six-blade disc turbine on a central shaft."""
    vessel, height, shaft = 0.4, 0.9, 0.02
    radius, clearance = design["impeller_diameter_m"] / 2, design["impeller_height_m"]
    omega = design["speed_rad_s"]
    blade_inner, blade_height, thickness = 0.62 * radius, 0.4 * radius, 0.012
    r = np.unique(
        np.concatenate(
            [
                np.linspace(shaft, blade_inner, level.cells(6) + 1),
                np.linspace(blade_inner, radius, level.cells(8) + 1),
                np.linspace(radius, vessel, level.cells(14) + 1),
            ]
        )
    )
    z = np.unique(
        np.concatenate(
            [
                np.linspace(0, clearance - blade_height / 2, level.cells(8) + 1),
                np.linspace(clearance - blade_height / 2, clearance - thickness / 2,
                            level.cells(4) + 1),
                np.linspace(clearance + thickness / 2, clearance + blade_height / 2,
                            level.cells(4) + 1),
                np.linspace(clearance + blade_height / 2, height, level.cells(16) + 1),
            ]
        )
    )  # fmt: skip
    per_blade = level.cells(12, 8)
    theta = np.linspace(0, 2 * pi, 6 * per_blade + 1)
    rc, zc = 0.5 * (r[1:] + r[:-1]), 0.5 * (z[1:] + z[:-1])
    shape = (len(rc), len(theta) - 1, len(zc))
    rr = np.broadcast_to(rc[:, None, None], shape)
    zz = np.broadcast_to(zc[None, None, :], shape)
    sector = np.broadcast_to((np.arange(len(theta) - 1) % per_blade)[None, :, None], shape)
    disc = (np.abs(zz - clearance) < thickness / 2) & (rr < 0.78 * radius)
    blades = (
        (sector == 0)
        & (rr > blade_inner)
        & (rr < radius)
        & (np.abs(zz - clearance) < blade_height / 2)
    )
    rotating = (rr < radius + 0.04) & (np.abs(zz - clearance) < blade_height / 2 + 0.04)
    mesh = hex_mesh()
    mesh.add_block(
        _polar_grid(r, theta, z),
        {"i0": "shaft", "i1": "tank", "k0": "bottom", "k1": "lid"},
        keep=~(disc | blades),
        exposed="impeller",
        zone=np.where(rotating, "rotor", None).astype(object),
    )
    patches = _finish(
        case,
        mesh,
        {
            "shaft": rotating_wall(omega), "tank": wall(), "bottom": wall(), "lid": slip(),
            "impeller": wall(),
        },
    )  # fmt: skip
    functions = post.forces("impeller", ["impeller"], WATER["density"])
    functions |= {
        "speed": {"type": "mag", "libs": "(fieldFunctionObjects)", "field": "U",
                  "result": "magU", "writeControl": "none", "log": False}
    }  # fmt: skip
    functions |= post.volume_value("mean_speed", "volAverage", "magU")
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam",
            velocity=omega * radius,
            length=blade_height,
            iterations=level.iterations,
            closed=True,
            rotations=(rotation("rotor", omega, ("tank", "bottom", "lid")),),
            functions=functions,
            **WATER,
        ),
    )

    def measure(directory):
        torque = float(post.last_moment(directory, "impeller")[2])
        return {
            "shaft_power_w": -torque * omega,
            "mean_speed_m_s": post.last_volume(directory, "mean_speed"),
        }

    return run_plan(solve=[["simpleFoam"]], measure=measure)


CASE = _case(
    "rushton_impeller",
    "Rushton turbine in a stirred vessel",
    rushton_impeller,
    (
        variable("impeller_diameter_m", 0.22, 0.32, 0.267),
        variable("impeller_height_m", 0.2, 0.4, 0.27),
        variable("speed_rad_s", 7.0, 16.0, 10.5),
    ),
    "shaft_power_w",
    (at_least("mean_speed_m_s"),),
    positive=("shaft_power_w",),
    simplification="single phase, flat bottom",
)
