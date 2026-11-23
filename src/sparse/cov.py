"""gpml covariance functions with gpml's hyperparameter layout.

Hyperparameters are ``log`` values exactly as in gpml, so a gpml ``hyp.cov`` vector can be
used unchanged:

* ``SEard``       ``[log ell_1 .. log ell_D, log sf]``     (``covSEard``)
* ``SEiso``       ``[log ell, log sf]``                      (``covSEiso``)
* ``Maternard``   ``[log ell_1 .. log ell_D, log sf]``     (``covMaternard``, degree 1, 3 or 5)
* ``Materniso``   ``[log ell, log sf]``                      (``covMaterniso``)
* ``ADD``         ``[log ell_1 .. log ell_D, log sf_r (r in orders)]``
                  (``covADD`` with ``covSEisoU`` per dimension, Duvenaud et al. 2011):
                  ``k = sum_r sf_r^2 e_r(k_1, .., k_D)``, ``e_r`` the ``r``-th elementary
                  symmetric polynomial of the 1-D squared-exponential kernels.

Calling convention follows gpml: ``cov(hyp, x)`` is ``K(x, x)``, ``cov(hyp, x, z)`` is
``K(x, z)`` and ``cov(hyp, x, "diag")`` is ``diag K(x, x)``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_KINDS = ("SEard", "SEiso", "Maternard", "Materniso", "ADD")


def _safe_sqrt(s: Tensor) -> Tensor:
    pos = s > 0
    return torch.where(
        pos, torch.sqrt(torch.where(pos, s, torch.ones_like(s))), torch.zeros_like(s)
    )


class Cov:
    """A gpml covariance function."""

    def __init__(self, kind: str = "SEard", degree: int = 3, orders: int | list[int] = 2):
        """``degree`` is the Matern degree; ``orders`` (``ADD`` only) is the maximum
        interaction order ``R`` (meaning ``1..R``) or an explicit list such as ``[1, 3]``."""
        self.orders = list(range(1, orders + 1)) if isinstance(orders, int) else list(orders)
        if kind == "ADD" and (not self.orders or min(self.orders) < 1):
            raise ValueError("ADD needs interaction orders >= 1")
        if kind not in _KINDS:
            raise ValueError(f"unknown covariance {kind!r}; choose from {_KINDS}")
        if kind.startswith("Matern") and degree not in (1, 3, 5):
            raise ValueError("Matern degree must be 1, 3 or 5")
        self.kind, self.degree = kind, degree

    def num_hyp(self, dim: int) -> int:
        if self.kind == "ADD":
            return dim + len(self.orders)
        return dim + 1 if self.kind.endswith("ard") else 2

    def _additive(self, hyp: Tensor, x: Tensor, z: Tensor | str | None) -> Tensor:
        D, R = x.shape[1], max(self.orders)
        if R > D:
            raise ValueError(f"interaction order {R} exceeds the input dimension {D}")
        ell, sf2 = torch.exp(hyp[:D]), torch.exp(2 * hyp[D:])
        if isinstance(z, str):  # diag: every 1-D kernel is 1, e_r = C(D, r)
            counts = [float(math.comb(D, r)) for r in self.orders]
            return sum(s2 * c for s2, c in zip(sf2, counts, strict=True)) * torch.ones(
                x.shape[0], dtype=x.dtype
            )
        a = x / ell
        b = a if z is None else z / ell
        kd = torch.exp(-0.5 * (a.unsqueeze(-2) - b.unsqueeze(-3)).pow(2))  # (n, m, D)
        power = [(kd**k).sum(-1) for k in range(1, R + 1)]  # power sums p_k
        e = [torch.ones_like(power[0])]
        for r in range(1, R + 1):  # Newton-Girard (gpml elsympol)
            e.append(
                torch.stack(
                    [(-1) ** (k - 1) * e[r - k] * power[k - 1] for k in range(1, r + 1)]
                ).sum(0)
                / r
            )
        return torch.stack([sf2[i] * e[r] for i, r in enumerate(self.orders)]).sum(0)

    def __call__(self, hyp: Tensor, x: Tensor, z: Tensor | str | None = None) -> Tensor:
        if self.kind == "ADD":
            return self._additive(hyp, x, z)
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
