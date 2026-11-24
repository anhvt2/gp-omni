"""Compatibility imports for the phase-based GP implementation."""

from gp_omni.build.gpml import ProductKernel, gpml_kernel, gpml_likelihood, gpml_mean

__all__ = ["ProductKernel", "gpml_kernel", "gpml_likelihood", "gpml_mean"]
