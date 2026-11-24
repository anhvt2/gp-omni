"""Compatibility imports for the renamed Gaussian-process surrogates."""

from gp_omni.sparse.gpc import GPC as OmniGPC
from gp_omni.sparse.gpr import GPR as OmniGP

__all__ = ["OmniGP", "OmniGPC"]
