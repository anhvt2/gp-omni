"""Linear elastic statics and vibration with quadratic tetrahedra.

Units are mm, N, MPa, tonne and seconds, so densities are tonne/mm^3.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTetP2,
    ElementVector,
    FacetBasis,
    LinearForm,
    MeshTet,
)
from skfem.helpers import dot
from skfem.models.elasticity import lame_parameters, linear_elasticity

from .facets import area, surface_integral
from .linear_solver import factorization

ELEMENT = ElementVector(ElementTetP2())
# exact on straight-sided tetrahedra: quadratic stiffness and quartic mass integrands
STIFFNESS_ORDER, MASS_ORDER = 2, 4
STRESS_ORDER = 6

fixed_list = Sequence[tuple[np.ndarray, Sequence[int]]]  # (facets, fixed components 0..2)
traction_fn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (x, normal) -> traction


@dataclass(frozen=True)
class material:
    young_mpa: float
    poisson: float
    density_t_mm3: float


@dataclass(frozen=True)
class load:
    facets: np.ndarray
    traction: traction_fn


def total_force(
    mesh: MeshTet,
    facets: np.ndarray,
    force: Sequence[float],
    weight: Callable[[np.ndarray], np.ndarray] | None = None,
) -> load:
    """A traction with resultant ``force``, uniform or proportional to ``weight(x) >= 0``."""
    total = area(mesh, facets) if weight is None else surface_integral(mesh, facets, weight)
    if total <= 0:
        raise ValueError("the load weight integrates to zero")
    density = np.asarray(force, float) / total

    def traction(x, normal):
        shape = np.ones_like(x[0]) if weight is None else weight(x)
        return density[:, None, None] * shape

    return load(facets, traction)


def bearing_force(
    mesh: MeshTet,
    facets: np.ndarray,
    force: Sequence[float],
    center: Sequence[float],
    radius: float,
) -> load:
    """A pin load: cosine traction on the half of a hole that faces ``force``.

    ``force`` must be normal to the hole axis, which passes through ``center``.
    """
    direction = np.asarray(force, float) / np.linalg.norm(force)

    def weight(x):
        offset = x - np.reshape(center, (3,) + (1,) * (x.ndim - 1))
        return np.maximum(np.einsum("i,i...->...", direction, offset) / radius, 0)

    return total_force(mesh, facets, force, weight)


def clamp(facets: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Hold every displacement component on the facets."""
    return facets, (0, 1, 2)


def pressure(facets: np.ndarray, value_mpa: float) -> load:
    return load(facets, lambda x, normal: -value_mpa * normal)


@dataclass
class stress_field:
    """Von Mises stress sampled at quadrature points, with their volumes and positions."""

    values: np.ndarray
    weights: np.ndarray
    points: np.ndarray

    def percentile(
        self, percent: float, keep: Callable[[np.ndarray], np.ndarray] | None = None
    ) -> float:
        """Volume-weighted percentile, optionally restricted to ``keep(points)``."""
        mask = np.ones(self.values.shape, bool) if keep is None else keep(self.points)
        if not mask.any():
            raise ValueError("the stress region contains no quadrature point")
        values, weights = self.values[mask], self.weights[mask]
        order = np.argsort(values)
        cumulative = np.cumsum(weights[order])
        return float(np.interp(percent / 100 * cumulative[-1], cumulative, values[order]))


@dataclass
class static_result:
    basis: Basis
    displacement: np.ndarray
    solid_material: material


class structure:
    """A supported elastic solid; one factorization serves every load case and the modes."""

    def __init__(self, mesh: MeshTet, solid_material: material, fixed: fixed_list):
        self.material = solid_material
        self.basis = Basis(mesh, ELEMENT, intorder=STIFFNESS_ORDER)
        lam, mu = lame_parameters(solid_material.young_mpa, solid_material.poisson)
        stiffness = linear_elasticity(lam, mu).assemble(self.basis)
        dofs = [
            self.basis.get_dofs(facets=facets).all([f"u^{axis + 1}" for axis in components])
            for facets, components in fixed
        ]
        self.free = self.basis.complement_dofs(np.unique(np.concatenate(dofs)))
        self.factors = factorization(stiffness[self.free][:, self.free])

    def deform(self, loads: Sequence[load]) -> static_result:
        force = np.zeros(self.basis.N)
        for item in loads:
            surface = FacetBasis(self.basis.mesh, ELEMENT, facets=item.facets, intorder=6)
            form = LinearForm(lambda v, w, item=item: dot(item.traction(w.x, w.n), v))
            force += form.assemble(surface)
        displacement = np.zeros(self.basis.N)
        displacement[self.free] = self.factors.solve(force[self.free])
        return static_result(self.basis, displacement, self.material)

    def frequencies(
        self, count: int = 1, point_masses: Sequence[tuple[np.ndarray, float]] = ()
    ) -> np.ndarray:
        """Lowest natural frequencies in Hz; ``point_masses`` spread a mass (t) over facets."""
        density = self.material.density_t_mm3
        mesh = self.basis.mesh
        volume = Basis(mesh, ELEMENT, intorder=MASS_ORDER)
        mass = BilinearForm(lambda u, v, w: density * dot(u, v)).assemble(volume)
        for facets, tonnes in point_masses:
            surface = FacetBasis(mesh, ELEMENT, facets=facets, intorder=4)
            per_area = tonnes / area(mesh, facets)
            form = BilinearForm(lambda u, v, w, per_area=per_area: per_area * dot(u, v))
            mass = mass + form.assemble(surface)
        values = self.factors.smallest_eigenvalues(mass[self.free][:, self.free], count)
        return np.sqrt(values) / (2 * np.pi)


def solve_static(
    mesh: MeshTet, solid_material: material, fixed: fixed_list, loads: Sequence[load]
) -> static_result:
    return structure(mesh, solid_material, fixed).deform(loads)


def natural_frequencies(
    mesh: MeshTet,
    solid_material: material,
    fixed: fixed_list,
    count: int = 1,
    point_masses: Sequence[tuple[np.ndarray, float]] = (),
) -> np.ndarray:
    return structure(mesh, solid_material, fixed).frequencies(count, point_masses)


def mean_displacement(result: static_result, facets: np.ndarray) -> np.ndarray:
    """Area-averaged displacement vector over the facets, in mm."""
    surface = FacetBasis(result.basis.mesh, ELEMENT, facets=facets, intorder=4)
    values = np.asarray(surface.interpolate(result.displacement))
    return (values * surface.dx).sum(axis=(1, 2)) / surface.dx.sum()


def von_mises(result: static_result) -> stress_field:
    # sample closer to the element faces than the stiffness quadrature, where peaks sit
    basis = Basis(result.basis.mesh, ELEMENT, intorder=STRESS_ORDER)
    gradient = basis.interpolate(result.displacement).grad
    strain = (gradient + gradient.swapaxes(0, 1)) / 2
    _, mu = lame_parameters(result.solid_material.young_mpa, result.solid_material.poisson)
    # the volumetric part of the stress does not enter the von Mises stress
    deviator = strain - np.eye(3)[:, :, None, None] * np.trace(strain) / 3
    values = 2 * mu * np.sqrt(1.5 * (deviator**2).sum(axis=(0, 1)))
    return stress_field(values, basis.dx, np.asarray(basis.global_coordinates()))


def mass_kg(mesh: MeshTet, solid_material: material) -> float:
    return volume_mm3(mesh) * solid_material.density_t_mm3 * 1_000


def volume_mm3(mesh: MeshTet) -> float:
    corners = mesh.p[:, mesh.t]
    edges = (corners[:, 1:] - corners[:, :1]).transpose(2, 0, 1)
    return float(np.abs(np.linalg.det(edges)).sum() / 6)
