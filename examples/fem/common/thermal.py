"""Steady heat conduction with prescribed temperature, heat flux and convection.

Units are mm, W and K, so conductivity is W/(mm K) and film coefficients W/(mm^2 K).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTetP2,
    FacetBasis,
    Functional,
    LinearForm,
    MeshTet,
    condense,
)
from skfem.helpers import dot, grad

from .facets import area
from .linear_solver import factorization

ELEMENT = ElementTetP2()


@dataclass(frozen=True)
class heat_flux:
    facets: np.ndarray
    watts: float
    where: Callable[[np.ndarray], np.ndarray] | None = None


def power_input(
    facets: np.ndarray, watts: float, where: Callable[[np.ndarray], np.ndarray] | None = None
) -> heat_flux:
    """``watts`` spread uniformly over the facets, or over the patch ``where(x)``."""
    return heat_flux(facets, watts, where)


def rectangle(
    x_span: tuple[float, float], y_span: tuple[float, float]
) -> Callable[[np.ndarray], np.ndarray]:
    """Predicate for points whose x and y lie in the spans, e.g. under a chip."""

    def inside(p):
        return (p[0] >= x_span[0]) & (p[0] <= x_span[1]) & (p[1] >= y_span[0]) & (p[1] <= y_span[1])

    return inside


@dataclass
class temperature_field:
    basis: Basis
    values: np.ndarray
    convections: Sequence[tuple[np.ndarray, float, float]]

    def maximum(self) -> float:
        return float(self.values.max())

    def mean(self, facets: np.ndarray) -> float:
        surface = FacetBasis(self.basis.mesh, ELEMENT, facets=facets, intorder=4)
        values = np.asarray(surface.interpolate(self.values))
        return float((values * surface.dx).sum() / surface.dx.sum())

    def convected_power(self) -> float:
        """Heat leaving through all convective surfaces, in W."""
        return sum(
            h * (self.mean(facets) - ambient) * area(self.basis.mesh, facets)
            for facets, h, ambient in self.convections
        )


def solve_conduction(
    mesh: MeshTet,
    conductivity: float,
    temperatures: Sequence[tuple[np.ndarray, float]] = (),
    fluxes: Sequence[heat_flux] = (),
    convections: Sequence[tuple[np.ndarray, float, float]] = (),
) -> temperature_field:
    """Solve for temperature; ``convections`` holds ``(facets, h, ambient temperature)``."""
    if not temperatures and not convections:
        raise ValueError("give a prescribed temperature or a convective surface")
    basis = Basis(mesh, ELEMENT, intorder=4)
    conduction = BilinearForm(lambda u, v, w: conductivity * dot(grad(u), grad(v)))
    matrix = conduction.assemble(basis)
    load = np.zeros(basis.N)
    for facets, h, ambient in convections:
        surface = FacetBasis(mesh, ELEMENT, facets=facets, intorder=4)
        matrix = matrix + BilinearForm(lambda u, v, w, h=h: h * u * v).assemble(surface)
        load += LinearForm(lambda v, w, q=h * ambient: q * v).assemble(surface)
    for flux in fluxes:
        load += _flux_load(mesh, flux)

    values = np.zeros(basis.N)
    fixed = []
    for facets, temperature in temperatures:
        dofs = basis.get_dofs(facets=facets).all()
        values[dofs] = temperature
        fixed.append(dofs)
    if fixed:
        reduced, vector, values, free = condense(matrix, load, x=values, D=np.concatenate(fixed))
        values[free] = factorization(reduced).solve(vector)
    else:
        values = factorization(matrix).solve(load)
    return temperature_field(basis, values, convections)


def _flux_load(mesh: MeshTet, flux: heat_flux) -> np.ndarray:
    surface = FacetBasis(mesh, ELEMENT, facets=flux.facets, intorder=8)

    def patch(x):
        return np.ones_like(x[0]) if flux.where is None else flux.where(x) * 1.0

    # Normalize with the same quadrature as the load so the total power is exact.
    covered = Functional(lambda w: patch(w.x)).assemble(surface)
    if covered <= 0:
        raise ValueError("the heat flux patch covers no quadrature point")
    form = LinearForm(lambda v, w: flux.watts / covered * patch(w.x) * v)
    return form.assemble(surface)
