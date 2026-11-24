"""Compatibility imports for the phase-based GP implementation."""

from gp_omni.train.gpytorch import fit_exact, fit_variational

__all__ = ["fit_exact", "fit_variational"]
