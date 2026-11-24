"""Compatibility imports for the phase-based GP implementation."""

from gp_omni.build.likelihood import Lik, LikErf, LikGauss, LikLogistic
from gp_omni.build.likelihood import make_likelihood as make_lik

__all__ = ["Lik", "LikErf", "LikGauss", "LikLogistic", "make_lik"]
