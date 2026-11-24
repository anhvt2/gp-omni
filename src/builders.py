"""Compatibility imports for the phase-based GP implementation."""

from gp_omni.build.gpytorch import (
    SVGP,
    ExactGPModel,
    MultitaskExactGP,
    additive_gp,
    exact_gp,
    kiss_gp,
    sgpr,
)

__all__ = ["SVGP", "ExactGPModel", "MultitaskExactGP", "additive_gp", "exact_gp", "kiss_gp", "sgpr"]
