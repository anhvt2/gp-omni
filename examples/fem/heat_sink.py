"""Aluminum pin-fin heat sink with a central heat source under its base.

The film coefficient is an assumed function of the gap between pins,
``h = H_MAX * gap / (gap + GAP_SCALE)``, which penalizes crowded pins. It is not a
validated convection correlation.

Run ``python examples/fem/heat_sink.py {describe,baseline,optimize}``.
"""

import numpy as np
from common import cli, solid, thermal
from common.facets import on_plane
from common.materials import ALUMINUM, ALUMINUM_CONDUCTIVITY_W_MM_K
from common.mesh import build_mesh
from common.problem import at_most, design_problem, variable

SIDE = 40.0
PINS_PER_SIDE = 5
SOURCE_SIDE, POWER_W = 16.0, 15.0
H_MAX_W_MM2_K, GAP_SCALE_MM = 60e-6, 1.5


def geometry(occ, design):
    base, height = design["base_thickness_mm"], design["pin_height_mm"]
    radius, pitch = design["pin_diameter_mm"] / 2, design["pitch_mm"]
    offsets = (np.arange(PINS_PER_SIDE) - (PINS_PER_SIDE - 1) / 2) * pitch + SIDE / 2
    plate = [(3, occ.addBox(0, 0, 0, SIDE, SIDE, base))]
    pins = [
        (3, occ.addCylinder(x, y, base / 2, 0, 0, height + base / 2, radius))
        for x in offsets
        for y in offsets
    ]
    occ.fuse(plate, pins)


def prepare_mesh(design, mesh_scale):
    return build_mesh(lambda occ: geometry(occ, design), 2.5 * mesh_scale, per_circle=8)


def solve(design, mesh_scale):
    return solve_mesh(design, prepare_mesh(design, mesh_scale))


def solve_mesh(design, mesh, fields=None):
    bottom = on_plane(mesh, 2, 0)
    exposed = np.setdiff1d(mesh.boundary_facets(), bottom)
    gap = design["pitch_mm"] - design["pin_diameter_mm"]
    film = H_MAX_W_MM2_K * gap / (gap + GAP_SCALE_MM)

    span = ((SIDE - SOURCE_SIDE) / 2, (SIDE + SOURCE_SIDE) / 2)
    temperature = thermal.solve_conduction(
        mesh,
        ALUMINUM_CONDUCTIVITY_W_MM_K,
        fluxes=[thermal.power_input(bottom, POWER_W, thermal.rectangle(span, span))],
        convections=[(exposed, film, 0.0)],
    )
    if fields is not None:
        fields.update(temperature_rise=temperature)
    return {
        "temperature_rise_k": temperature.maximum(),
        "mass_kg": solid.mass_kg(mesh, ALUMINUM),
        "convected_w": temperature.convected_power(),
    }


PROBLEM = design_problem(
    name="heat_sink",
    title="Pin-fin heat sink",
    variables=(
        variable("pin_height_mm", 8, 35, 20),
        variable("pin_diameter_mm", 1.5, 4.5, 3),
        variable("base_thickness_mm", 1.5, 6, 3),
        variable("pitch_mm", 5, 8.5, 7),
    ),
    objective="temperature_rise_k",
    constraints=(at_most("mass_kg", 0.025),),
    solve=solve,
)

if __name__ == "__main__":
    cli.main(PROBLEM)
