"""Kernels GPyTorch does not (non-deprecated) provide."""

from __future__ import annotations

import torch
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from torch import Tensor


class AdditiveKernel(Kernel):
    r"""gpml ``covADD``: ``k(x, x') = sum_{r in orders} s_r^2 e_r(k_1, ..., k_D)``.

    ``k_i`` is a squared-exponential kernel on input dimension ``i`` and ``e_r`` is the
    ``r``-th elementary symmetric polynomial of ``(k_1, ..., k_D)``, evaluated with the
    Newton-Girard identities in O(D * R). Orders are ``1..max_degree`` (order 1 = purely
    additive; higher orders add interactions between ``r`` distinct dimensions).

    As in gpml the length scale is shared across dimensions unless ``ard=True``. Each order
    has its own learned output scale. Replaces GPyTorch's deprecated
    ``NewtonGirardAdditiveKernel``.
    """

    has_lengthscale = True

    def __init__(self, num_dims: int, max_degree: int = 2, ard: bool = False, **kwargs):
        super().__init__(ard_num_dims=num_dims if ard else None, **kwargs)
        if not 1 <= max_degree <= num_dims:
            raise ValueError("need 1 <= max_degree <= num_dims")
        self.num_dims, self.max_degree = num_dims, max_degree
        self.register_parameter("raw_order_scale", torch.nn.Parameter(torch.zeros(max_degree)))
        self.register_constraint("raw_order_scale", Positive())

    @property
    def order_scale(self) -> Tensor:
        return self.raw_order_scale_constraint.transform(self.raw_order_scale)

    @order_scale.setter
    def order_scale(self, value) -> None:
        value = torch.as_tensor(value, dtype=self.raw_order_scale.dtype)
        self.initialize(raw_order_scale=self.raw_order_scale_constraint.inverse_transform(value))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        ell = self.lengthscale  # (..., 1, 1) shared or (..., 1, D)
        diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) / ell.unsqueeze(-2)
        ki = torch.exp(-0.5 * diff**2)  # (..., n, m, D)
        power = [(ki**k).sum(-1) for k in range(1, self.max_degree + 1)]  # power[k-1] = p_k
        e = [torch.ones_like(power[0])]
        for n in range(1, self.max_degree + 1):  # Newton-Girard
            terms = [(-1) ** (k - 1) * e[n - k] * power[k - 1] for k in range(1, n + 1)]
            e.append(torch.stack(terms).sum(0) / n)
        out = torch.stack([self.order_scale[r - 1] * e[r] for r in range(1, self.max_degree + 1)])
        out = out.sum(0)
        return out.diagonal(dim1=-2, dim2=-1) if diag else out
