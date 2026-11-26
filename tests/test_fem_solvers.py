"""Closed-form checks of the 3D FEM solvers used by ``examples/fem``."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skfem")
pytest.importorskip("gmsh")

from skfem import MeshTet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples/fem"))

from common import solid, thermal
from common.facets import area, on_cylinder, on_plane, select_facets
from common.mesh import build_mesh

STEEL = solid.material(young_mpa=210_000, poisson=0.3, density_t_mm3=7.85e-9)


def box(lx, ly, lz, nx, ny, nz):
    return MeshTet.init_tensor(
        np.linspace(0, lx, nx + 1), np.linspace(0, ly, ny + 1), np.linspace(0, lz, nz + 1)
    )


def plane(mesh, axis, value):
    return select_facets(mesh, lambda p: np.abs(p[axis] - value) < 1e-6)


def test_select_facets_returns_one_boundary_plane():
    mesh = box(4, 2, 1, 4, 2, 1)
    facets = plane(mesh, 0, 4)
    assert len(facets) == 2 * 2 * 1
    assert np.allclose(mesh.p[0, mesh.facets[:, facets]], 4)


def test_select_facets_rejects_an_empty_selection():
    with pytest.raises(ValueError, match="no boundary facet"):
        plane(box(1, 1, 1, 1, 1, 1), 0, 5)


def test_uniaxial_tension_is_exact():
    mesh = box(100, 10, 10, 5, 1, 1)
    fixed = [(plane(mesh, axis, 0), (axis,)) for axis in range(3)]
    loads = [solid.total_force(mesh, plane(mesh, 0, 100), (10_000, 0, 0))]
    result = solid.solve_static(mesh, STEEL, fixed, loads)

    end = plane(mesh, 0, 100)
    assert solid.mean_displacement(result, end)[0] == pytest.approx(100 * 100 / 210_000, rel=1e-5)
    stress = solid.von_mises(result)
    assert stress.values.min() == pytest.approx(100, rel=1e-4)
    assert stress.values.max() == pytest.approx(100, rel=1e-4)
    assert stress.weights.sum() == pytest.approx(100 * 10 * 10)


def test_hydrostatic_pressure_gives_no_deviatoric_stress():
    mesh = box(10, 10, 10, 2, 2, 2)
    fixed = [(plane(mesh, axis, 0), (axis,)) for axis in range(3)]
    loads = [solid.pressure(plane(mesh, axis, 10), 50.0) for axis in range(3)]
    result = solid.solve_static(mesh, STEEL, fixed, loads)

    expected = -50 * (1 - 2 * 0.3) / 210_000 * 10
    assert solid.mean_displacement(result, plane(mesh, 0, 10))[0] == pytest.approx(
        expected, rel=1e-5
    )
    assert solid.von_mises(result).values.max() < 1e-3


def test_cantilever_deflection_matches_timoshenko_theory():
    length, width, height, force = 200.0, 10.0, 20.0, 1_000.0
    mesh = box(length, width, height, 40, 2, 4)
    fixed = [(plane(mesh, 0, 0), (0, 1, 2))]
    tip = plane(mesh, 0, length)
    result = solid.solve_static(mesh, STEEL, fixed, [solid.total_force(mesh, tip, (0, 0, -force))])

    inertia = width * height**3 / 12
    shear = 210_000 / 2.6
    expected = force * length**3 / (3 * 210_000 * inertia) + force * length / (
        5 / 6 * shear * width * height
    )
    assert -solid.mean_displacement(result, tip)[2] == pytest.approx(expected, rel=0.02)


def test_cantilever_first_frequency_matches_beam_theory():
    length, width, height = 200.0, 10.0, 20.0
    mesh = box(length, width, height, 40, 2, 4)
    fixed = [(plane(mesh, 0, 0), (0, 1, 2))]
    first = solid.natural_frequencies(mesh, STEEL, fixed, count=1)[0]

    inertia = height * width**3 / 12  # weak axis
    expected = (
        1.875104**2
        / (2 * np.pi * length**2)
        * np.sqrt(210_000 * inertia / (7.85e-9 * width * height))
    )
    assert first == pytest.approx(expected, rel=0.02)


def test_tip_mass_lowers_frequency_as_rayleigh_predicts():
    length, width, height, tip_mass_t = 200.0, 10.0, 20.0, 5e-4
    mesh = box(length, width, height, 40, 2, 4)
    fixed = [(plane(mesh, 0, 0), (0, 1, 2))]
    masses = [(plane(mesh, 0, length), tip_mass_t)]
    first = solid.natural_frequencies(mesh, STEEL, fixed, count=1, point_masses=masses)[0]

    stiffness = 3 * 210_000 * height * width**3 / 12 / length**3
    beam_mass = 7.85e-9 * length * width * height
    expected = np.sqrt(stiffness / (tip_mass_t + 0.2427 * beam_mass)) / (2 * np.pi)
    assert first == pytest.approx(expected, rel=0.03)


def test_mass_is_density_times_volume():
    assert solid.mass_kg(box(100, 10, 10, 2, 1, 1), STEEL) == pytest.approx(0.0785)


def test_stress_percentile_can_exclude_a_region():
    mesh = box(100, 10, 20, 20, 1, 2)
    fixed = [(plane(mesh, 0, 0), (0, 1, 2))]
    load = solid.total_force(mesh, plane(mesh, 0, 100), (0, 0, -100))
    stress = solid.von_mises(solid.solve_static(mesh, STEEL, fixed, [load]))
    everywhere = stress.percentile(99)
    away_from_clamp = stress.percentile(99, keep=lambda p: p[0] > 20)
    assert away_from_clamp < everywhere


def test_prescribed_heat_flux_gives_a_linear_temperature():
    mesh = box(50, 10, 10, 5, 1, 1)
    temperature = thermal.solve_conduction(
        mesh,
        conductivity=0.2,
        temperatures=[(plane(mesh, 0, 0), 0.0)],
        fluxes=[thermal.power_input(plane(mesh, 0, 50), 20.0)],
    )
    assert temperature.maximum() == pytest.approx(20 / 100 * 50 / 0.2, rel=1e-6)


def test_convection_balances_conduction():
    mesh = box(50, 10, 10, 5, 1, 1)
    end = plane(mesh, 0, 50)
    temperature = thermal.solve_conduction(
        mesh,
        conductivity=0.2,
        temperatures=[(plane(mesh, 0, 0), 100.0)],
        convections=[(end, 0.001, 20.0)],
    )
    expected = (0.2 / 50 * 100 + 0.001 * 20) / (0.2 / 50 + 0.001)
    assert temperature.mean(end) == pytest.approx(expected, rel=1e-6)
    assert temperature.convected_power() == pytest.approx(0.001 * (expected - 20) * 100, rel=1e-6)


def test_patch_power_is_exact_and_leaves_through_convection():
    mesh = box(40, 40, 4, 8, 8, 1)
    top, bottom = plane(mesh, 2, 4), plane(mesh, 2, 0)
    patch = thermal.power_input(bottom, 15.0, where=lambda x: (x[0] < 17) & (x[1] < 13))
    temperature = thermal.solve_conduction(
        mesh, conductivity=0.2, fluxes=[patch], convections=[(top, 0.0005, 0.0)]
    )
    assert temperature.convected_power() == pytest.approx(15.0, rel=1e-6)


def test_on_plane_selects_the_face_at_a_coordinate():
    mesh = box(4, 2, 1, 4, 2, 1)
    assert np.array_equal(on_plane(mesh, 0, 4), plane(mesh, 0, 4))


def test_on_cylinder_selects_the_curved_surface_only():
    def ring(occ):
        outer = occ.addCylinder(5, 0, 0, 0, 0, 4, 10)
        inner = occ.addCylinder(5, 0, -1, 0, 0, 6, 4)
        occ.cut([(3, outer)], [(3, inner)])

    mesh = build_mesh(ring, 1.5)
    bore = on_cylinder(mesh, axis=2, center=(5, 0, 0), radius=4)
    assert area(mesh, bore) == pytest.approx(2 * np.pi * 4 * 4, rel=0.01)
    half = on_cylinder(mesh, axis=2, center=(5, 0, 0), radius=4, keep=lambda p: p[1] > -1e-6)
    assert area(mesh, half) == pytest.approx(np.pi * 4 * 4, rel=0.1)


def test_bearing_force_pushes_only_the_loaded_half_of_a_hole():
    def ring(occ):
        outer = occ.addCylinder(0, 0, 0, 0, 0, 4, 10)
        inner = occ.addCylinder(0, 0, -1, 0, 0, 6, 4)
        occ.cut([(3, outer)], [(3, inner)])

    mesh = build_mesh(ring, 1.5)
    bore = on_cylinder(mesh, axis=2, center=(0, 0, 0), radius=4)
    load = solid.bearing_force(mesh, bore, (100, 0, 0), center=(0, 0, 0), radius=4)
    x = np.array([[4.0, -4.0], [0.0, 0.0], [2.0, 2.0]])[:, :, None]
    traction = load.traction(x, None)
    assert traction[0, 0, 0] > 0
    assert np.all(traction[:, 1, 0] == 0)


def test_rectangle_marks_points_inside_a_patch():
    inside = thermal.rectangle((0, 2), (1, 3))
    points = np.array([[1.0, 3.0, 2.0], [2.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    assert inside(points).tolist() == [True, False, True]
