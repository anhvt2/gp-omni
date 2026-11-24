"""Triangular solves shared by GP construction and evaluation."""

import torch
from torch import Tensor


def _solve_lower(L: Tensor, b: Tensor) -> Tensor:
    return torch.linalg.solve_triangular(L, b, upper=False)


def _solve_upper_t(L: Tensor, b: Tensor) -> Tensor:
    """Solve ``L^T x = b`` for lower-triangular ``L``."""
    return torch.linalg.solve_triangular(L.transpose(-1, -2), b, upper=True)
