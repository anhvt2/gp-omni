"""Handbook room-temperature properties in mm, N, tonne units."""

from .solid import material

ALUMINUM = material(young_mpa=69_000, poisson=0.33, density_t_mm3=2.7e-9)
STEEL = material(young_mpa=210_000, poisson=0.3, density_t_mm3=7.85e-9)
ALUMINUM_CONDUCTIVITY_W_MM_K = 0.2
