"""gpml covariance functions with gpml's hyperparameter layout.

Hyperparameters are ``log`` values exactly as in gpml, so a gpml ``hyp.cov`` vector can be
used unchanged:

* ``SEard``       ``[log ell_1 .. log ell_D, log sf]``     (``covSEard``)
* ``SEiso``       ``[log ell, log sf]``                      (``covSEiso``)
* ``Maternard``   ``[log ell_1 .. log ell_D, log sf]``     (``covMaternard``, degree 1, 3 or 5)
* ``Materniso``   ``[log ell, log sf]``                      (``covMaterniso``)

Calling convention follows gpml: ``cov(hyp, x)`` is ``K(x, x)``, ``cov(hyp, x, z)`` is
``K(x, z)`` and ``cov(hyp, x, "diag")`` is ``diag K(x, x)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

_KINDS = ("SEard", "SEiso", "Maternard", "Materniso")


def _safe_sqrt(s: Tensor) -> Tensor:
    pos = s > 0
    return torch.where(
        pos, torch.sqrt(torch.where(pos, s, torch.ones_like(s))), torch.zeros_like(s)
    )


class Cov:
    """A gpml covariance function."""

    def __init__(self, kind: str = "SEard", degree: int = 3):
        if kind not in _KINDS:
            raise ValueError(f"unknown covariance {kind!r}; choose from {_KINDS}")
        if kind.startswith("Matern") and degree not in (1, 3, 5):
            raise ValueError("Matern degree must be 1, 3 or 5")
        self.kind, self.degree = kind, degree

    def num_hyp(self, dim: int) -> int:
        return dim + 1 if self.kind.endswith("ard") else 2

    def __call__(self, hyp: Tensor, x: Tensor, z: Tensor | str | None = None) -> Tensor:
        sf2 = torch.exp(2 * hyp[-1])
        if isinstance(z, str):
            if z != "diag":
                raise ValueError("z must be a tensor, None or 'diag'")
            return sf2 * torch.ones(x.shape[0], dtype=x.dtype)
        ell = torch.exp(hyp[:-1])  # (D,) for ard, (1,) for iso
        a = x / ell
        b = a if z is None else z / ell
        d2 = (a.unsqueeze(-2) - b.unsqueeze(-3)).pow(2).sum(-1)
        if self.kind.startswith("SE"):
            return sf2 * torch.exp(-0.5 * d2)
        d = self.degree
        t = (d**0.5) * _safe_sqrt(d2)
        f = {1: torch.ones_like(t), 3: 1 + t, 5: 1 + t + t**2 / 3}[d]
        return sf2 * f * torch.exp(-t)
