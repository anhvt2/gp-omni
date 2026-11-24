"""Stationary correlation functions with the co-kriging hyperparameterisation.

Hyperparameters ``hp`` are ``log10`` of per-dimension precisions ``ell`` (``ell = 10**hp``),
so ``corrgauss(d) = exp(-sum(ell * d**2))``. A length-1 ``hp`` is broadcast over all
dimensions (isotropic).
"""

from __future__ import annotations

import torch
from torch import Tensor

CORRELATIONS = ("gauss", "exp", "matern32", "matern52")


def _safe_sqrt(s: Tensor) -> Tensor:
    """sqrt with a finite gradient at 0."""
    pos = s > 0
    return torch.where(
        pos, torch.sqrt(torch.where(pos, s, torch.ones_like(s))), torch.zeros_like(s)
    )


def correlation(name: str, hp: Tensor, x1: Tensor, x2: Tensor) -> Tensor:
    """Correlation matrix ``Psi(x1, x2)`` of shape ``(..., n1, n2)``."""
    d = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., n1, n2, dim)
    ell = torch.pow(10.0, hp)
    if name == "gauss":
        return torch.exp(-(d**2 * ell).sum(-1))
    if name == "exp":
        return torch.exp(-(d.abs() * ell).sum(-1))
    if name in ("matern32", "matern52"):
        dist = _safe_sqrt((d**2 * ell).sum(-1))
        if name == "matern32":
            t = 3.0**0.5 * dist
            return (1.0 + t) * torch.exp(-t)
        t = 5.0**0.5 * dist
        return (1.0 + t + t**2 / 3.0) * torch.exp(-t)
    raise ValueError(f"unknown correlation {name!r}; choose from {CORRELATIONS}")
