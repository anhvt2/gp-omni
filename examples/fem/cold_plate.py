"""Aluminum cold plate with two straight water channels under two chips.

Each channel carries a fixed flow. The film coefficient follows Dittus-Boelter and the
pressure drop Blasius; the water temperature rise and manifolds are not modeled.

Run ``python examples/fem/cold_plate.py {describe,baseline,optimize}``.
"""

import numpy as np
from common import cli, solid, thermal
from common.facets import on_cylinder, on_plane
from common.materials import ALUMINUM, ALUMINUM_CONDUCTIVITY_W_MM_K
from common.mesh import build_mesh
from common.problem import at_most, design_problem, variable

LENGTH, WIDTH = 100.0, 60.0
CHIPS = (((20, 40), (10, 30), 60.0), ((62, 77), (35, 50), 30.0))  # x span, y span, watts
FLOW_M3_S = 1.5e-3 / 60  # per channel
DENSITY, VISCOSITY = 997.0, 0.00089  # water, kg/m^3 and Pa s
CONDUCTIVITY, PRANDTL = 0.6, 6.1  # water, W/(m K)


def channel_flow(diameter_mm: float) -> tuple[float, float]:
    """Film coefficient in W/(mm^2 K) and pressure drop in kPa of one channel."""
    diameter = diameter_mm / 1_000
    velocity = FLOW_M3_S / (np.pi * diameter**2 / 4)
    reynolds = DENSITY * velocity * diameter / VISCOSITY
    nusselt = 0.023 * reynolds**0.8 * PRANDTL**0.4
    friction = 0.316 * reynolds**-0.25
    drop = friction * LENGTH / 1_000 / diameter * DENSITY * velocity**2 / 2
    return float(nusselt * CONDUCTIVITY / diameter * 1e-6), float(drop / 1_000)


def channel_axes(design) -> list[tuple[float, float]]:
    """(y, z) of both channels; ``depth_fraction`` 1 puts them 1 mm under the chip face."""
    diameter, thickness = design["channel_diameter_mm"], design["thickness_mm"]
    z = diameter / 2 + 1 + design["depth_fraction"] * (thickness - diameter - 2)
    return [(design["first_channel_y_mm"], z), (design["second_channel_y_mm"], z)]


def geometry(occ, design):
    radius = design["channel_diameter_mm"] / 2
    plate = [(3, occ.addBox(0, 0, 0, LENGTH, WIDTH, design["thickness_mm"]))]
    channels = [
        (3, occ.addCylinder(-1, y, z, LENGTH + 2, 0, 0, radius)) for y, z in channel_axes(design)
    ]
    occ.cut(plate, channels)


def prepare_mesh(design, mesh_scale):
    return build_mesh(lambda occ: geometry(occ, design), 3.5 * mesh_scale)


def solve(design, mesh_scale):
    return solve_mesh(design, prepare_mesh(design, mesh_scale))


def solve_mesh(design, mesh, fields=None):
    radius, thickness = design["channel_diameter_mm"] / 2, design["thickness_mm"]
    top = on_plane(mesh, 2, thickness)
    wetted = np.union1d(*(on_cylinder(mesh, 0, (0, y, z), radius) for y, z in channel_axes(design)))
    film, drop = channel_flow(design["channel_diameter_mm"])
    chips = [
        thermal.power_input(top, watts, thermal.rectangle(along, across))
        for along, across, watts in CHIPS
    ]
    temperature = thermal.solve_conduction(
        mesh, ALUMINUM_CONDUCTIVITY_W_MM_K, fluxes=chips, convections=[(wetted, film, 0.0)]
    )
    if fields is not None:
        fields.update(temperature_rise=temperature)
    return {
        "temperature_rise_k": temperature.maximum(),
        "pressure_drop_kpa": drop,
        "mass_kg": solid.mass_kg(mesh, ALUMINUM),
        "convected_w": temperature.convected_power(),
    }


PROBLEM = design_problem(
    name="cold_plate",
    title="Liquid cold plate",
    variables=(
        variable("channel_diameter_mm", 3, 7, 5),
        variable("first_channel_y_mm", 10, 26, 18),
        variable("second_channel_y_mm", 34, 52, 42),
        variable("depth_fraction", 0, 1, 0.5),
        variable("thickness_mm", 10, 16, 12),
    ),
    objective="temperature_rise_k",
    constraints=(at_most("pressure_drop_kpa", 1.5), at_most("mass_kg", 0.2)),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
