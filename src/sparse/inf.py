"""Compatibility imports for the phase-based GP implementation."""

from gp_omni.infer.posterior import Posterior
from gp_omni.infer.posterior import gaussian_posterior as inf_gauss_lik
from gp_omni.infer.posterior import laplace_posterior as inf_laplace

__all__ = ["Posterior", "inf_gauss_lik", "inf_laplace"]
