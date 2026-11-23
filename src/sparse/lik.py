"""gpml likelihood functions: ``likGauss``, ``likErf`` (probit) and ``likLogistic``.

Each likelihood provides

* ``laplace(hyp, y, f)``: ``lp, dlp, d2lp`` of ``log p(y|f)`` w.r.t. ``f`` (gpml's
  ``inf = 'infLaplace'`` mode), kept differentiable so autograd can reach ``d3lp``;
* ``predict(hyp, y, mu, s2)``: ``lp, ymu, ys2`` (gpml's prediction mode, ``y`` may be ``None``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _derivs(lp_fn, f: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Value, first and second derivative of an elementwise ``lp_fn`` w.r.t. ``f``."""
    if not f.requires_grad:
        f = f.detach().requires_grad_(True)
    with torch.enable_grad():
        lp = lp_fn(f)
        (dlp,) = torch.autograd.grad(lp.sum(), f, create_graph=True)
        (d2lp,) = torch.autograd.grad(dlp.sum(), f, create_graph=True)
    return lp, dlp, d2lp


class Lik:
    kind = ""

    def num_hyp(self) -> int:
        return 0


class LikGauss(Lik):
    """Gaussian noise, ``hyp = [log sn]``."""

    kind = "likGauss"

    def num_hyp(self) -> int:
        return 1

    def laplace(self, hyp: Tensor, y: Tensor, f: Tensor):
        sn2 = torch.exp(2 * hyp[0])
        lp = -((y - f) ** 2) / (2 * sn2) - 0.5 * torch.log(2 * math.pi * sn2)
        return lp, (y - f) / sn2, -torch.ones_like(f) / sn2

    def noise_var(self, hyp: Tensor, x: Tensor) -> Tensor:
        """Per-sample noise variance (a constant for ``likGauss``)."""
        return torch.exp(2 * hyp[0]) * torch.ones(x.shape[0], dtype=x.dtype)

    def predict(
        self, hyp: Tensor, y: Tensor | None, mu: Tensor, s2: Tensor, x: Tensor | None = None
    ):
        sn2 = torch.exp(2 * hyp[0]) if x is None else self.noise_var(hyp, x)
        ys2 = s2 + sn2
        lp = (
            None if y is None else -((y - mu) ** 2) / (2 * ys2) - 0.5 * torch.log(2 * math.pi * ys2)
        )
        return lp, mu, ys2


class _Binary(Lik):
    """Shared pieces of the +/-1 label likelihoods."""

    def _logp(self, yf: Tensor) -> Tensor:  # log p(y|f) as a function of y*f
        raise NotImplementedError

    def laplace(self, hyp: Tensor, y: Tensor, f: Tensor):
        y = _labels(y, f)
        return _derivs(lambda t: self._logp(y * t), f)

    def _log_moment(self, y: Tensor, mu: Tensor, s2: Tensor) -> Tensor:
        """log of E_{f~N(mu,s2)}[p(y|f)] for ``s2 > 0``."""
        raise NotImplementedError

    def predict(
        self, hyp: Tensor, y: Tensor | None, mu: Tensor, s2: Tensor, x: Tensor | None = None
    ):
        yy = torch.ones_like(mu) if y is None else _labels(y, mu)
        if float(torch.linalg.norm(s2)) <= torch.finfo(mu.dtype).eps:
            lp = self._logp(yy * mu)
        else:
            lp = self._log_moment(yy, mu, s2)
        p = torch.exp(lp)
        return lp, 2 * p - 1, 4 * p * (1 - p)


def _labels(y: Tensor, like: Tensor) -> Tensor:
    y = y * torch.ones_like(like)
    return torch.where(y == 0, torch.ones_like(y), torch.sign(y))


class LikErf(_Binary):
    """Probit likelihood ``Phi(y f)``; the predictive moment is closed-form (as gpml)."""

    kind = "likErf"

    def _logp(self, yf: Tensor) -> Tensor:
        return torch.special.log_ndtr(yf)

    def _log_moment(self, y: Tensor, mu: Tensor, s2: Tensor) -> Tensor:
        return torch.special.log_ndtr(y * mu / torch.sqrt(1 + s2))


_LAM = math.sqrt(2.0) * torch.tensor([0.44, 0.41, 0.40, 0.39, 0.36])
_C = torch.tensor(
    [
        1.146480988574439e02,
        -1.508871030070582e03,
        2.676085036831241e03,
        -1.356294962039222e03,
        7.543285642111850e01,
    ]
)


class LikLogistic(_Binary):
    """Logistic likelihood ``1 / (1 + exp(-y f))``.

    The predictive moment follows gpml ``likLogistic`` (``'infEP'`` branch): a five-component
    scale mixture of probit terms, blended with a tail approximation (Nickisch).
    """

    kind = "likLogistic"

    def _logp(self, yf: Tensor) -> Tensor:
        return -torch.nn.functional.softplus(-yf)

    def _log_moment(self, y: Tensor, mu: Tensor, s2: Tensor) -> Tensor:
        z = (
            y[:, None]
            * mu[:, None]
            * _LAM[None, :]
            / torch.sqrt(1 + s2[:, None] * _LAM[None, :] ** 2)
        )
        lzc = torch.special.log_ndtr(z)
        mx = lzc.max(dim=1, keepdim=True).values
        lz = torch.log(torch.exp(lzc - mx) @ _C) + mx[:, 0]  # log(exp(A) * c)
        val = mu.abs() - 196.0 / 200.0 * s2 - 4.0  # empirically determined bound at val==0
        w = 1.0 / (1.0 + torch.exp(-10.0 * val))  # interpolation weights
        lz_tail = torch.minimum(s2 / 2 - mu.abs(), torch.full_like(mu, -0.1))
        agree = y * mu > 0  # label and mean agree: p(y|f) = 1 - p(-y|f)
        lz_tail = torch.where(agree, torch.log(1 - torch.exp(lz_tail)), lz_tail)
        return (1 - w) * lz + w * lz_tail


def make_lik(name: str) -> Lik:
    table = {"likGauss": LikGauss, "likErf": LikErf, "likLogistic": LikLogistic}
    if name not in table:
        raise ValueError(f"unknown likelihood {name!r}; choose from {tuple(table)}")
    return table[name]()
