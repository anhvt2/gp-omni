"""Polynomial regression bases (``regpoly0/1/2``)."""

from __future__ import annotations

from itertools import combinations_with_replacement

import torch
from torch import Tensor

REGRESSIONS = {"none": -1, "": -1, "regpoly0": 0, "regpoly1": 1, "regpoly2": 2}


def regression_matrix(kind: str | int, x: Tensor) -> Tensor:
    """Basis matrix ``F`` of shape ``(..., n, p)``.

    ``none`` gives ``p = 0`` (zero-mean process); degree ``k`` includes all monomials of
    total degree <= k.
    """
    degree = REGRESSIONS[kind] if isinstance(kind, str) else int(kind)
    n, dim = x.shape[-2], x.shape[-1]
    if degree < 0:
        return x.new_zeros(*x.shape[:-2], n, 0)
    cols = [x.new_ones(*x.shape[:-2], n)]
    for k in range(1, degree + 1):
        for idx in combinations_with_replacement(range(dim), k):
            col = x[..., idx[0]]
            for j in idx[1:]:
                col = col * x[..., j]
            cols.append(col)
    return torch.stack(cols, dim=-1)
