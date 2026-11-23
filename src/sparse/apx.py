"""Covariance approximations, a port of gpml's ``apx.m`` and ``apxSparse.m`` (v4.2).

Any sparse approximation to the posterior of a GP is equivalent to using the covariance

    Kt = Q + s * diag(g),   Q = Ku' inv(Kuu + diag(snu2)) Ku,   g = diag(K - Q)

where ``s`` in ``[0, 1]`` interpolates between FITC (``s = 1``, Snelson & Ghahramani 2006)
and the variational free energy bound VFE (``s = 0``, Titsias 2009), with sparse power EP
(Bui, Yan & Turner 2016) in between. All linear algebra is O(n nu^2) via the Woodbury
identity, following gpml's variable names (``V``, ``g``, ``Luu``, ``Lu``, ``LuV``, ``d``).

gpml also hand-codes the derivatives with respect to every hyperparameter
(``ldB2_deriv_sparse``); here they come from autograd, so the same code differentiates
w.r.t. covariance hyperparameters, inducing inputs ``xu`` and inducing noise ``snu``.

Conventions: gpml's ``chol`` returns an upper triangular factor, and so does everything
named ``L*`` below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from gp_omni.sparse.cov import Cov


def _tri_solve(R: Tensor, b: Tensor, upper: bool) -> Tensor:
    if b.ndim == 1:
        return torch.linalg.solve_triangular(R, b[:, None], upper=upper)[:, 0]
    return torch.linalg.solve_triangular(R, b, upper=upper)


@dataclass
class SparseCov:
    """gpml's ``{@apxSparse, cov, xu}``: a covariance plus (default) inducing inputs.

    As in gpml, a ``hyp['xu']`` passed to inference takes priority over ``xu``.
    """

    cov: Cov
    xu: Tensor | None = None


@dataclass
class Approx:
    """gpml's ``K`` structure: ``mvm``, ``P``, ``Pt`` and ``fun(W)``."""

    mvm: Callable[[Tensor], Tensor]
    P: Callable[[Tensor], Tensor]
    Pt: Callable[[Tensor], Tensor]
    fun: Callable[[Tensor], WFuns]
    sparse: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class WFuns:
    """Outputs of ``K.fun(W)``: ``ldB2 = log det(B)/2``, ``solveKiW(r) = (K + inv(W)) \\ r``
    and ``L`` (the matrix defining the predictive variance, see gpml ``gp.m``)."""

    ldB2: Tensor
    solveKiW: Callable[[Tensor], Tensor]
    L: Tensor


def _col(d: Tensor, r: Tensor) -> Tensor:
    return d.reshape(-1, *([1] * (r.ndim - 1))) * r


def apx(
    hyp: dict, cov: Cov | SparseCov, x: Tensor, opt: dict | None = None, want_L: bool = True
) -> Approx:
    """Build the covariance approximation: exact (dense) or sparse, like gpml's ``apx``."""
    opt = opt or {}
    if isinstance(cov, SparseCov):
        return _sparse(hyp, cov, x, float(opt.get("s", 1.0)), want_L)
    return _exact(hyp, cov, x)


# --------------------------------------------------------------------------- A) exact
def _exact(hyp: dict, cov: Cov, x: Tensor) -> Approx:
    K = cov(hyp["cov"], x)
    n = K.shape[0]

    def fun(W: Tensor) -> WFuns:
        if bool((W < 0).any()):
            raise NotImplementedError("exact inference with negative W (LU mode) is not ported")
        sW = torch.sqrt(W)
        L = torch.linalg.cholesky(
            torch.eye(n, dtype=K.dtype) + sW[:, None] * sW[None, :] * K, upper=True
        )
        ldB2 = torch.log(torch.diagonal(L)).sum()

        def solveKiW(r: Tensor) -> Tensor:
            u = _col(sW, r)
            return _col(sW, torch.cholesky_solve(u.reshape(n, -1), L, upper=True).reshape(r.shape))

        return WFuns(ldB2, solveKiW, L)

    ident = lambda v: v
    return Approx(lambda v: K @ v, ident, ident, fun, sparse=False)


# -------------------------------------------------------------------------- B) sparse
def _sparse(hyp: dict, sc: SparseCov, x: Tensor, s: float, want_L: bool) -> Approx:
    if not 0.0 <= s <= 1.0:
        raise ValueError("opt['s'] must lie in [0, 1]")
    xu = hyp.get("xu", sc.xu)  # hyp.xu has priority (gpml)
    if xu is None:
        raise ValueError("no inducing inputs: set SparseCov.xu or hyp['xu']")
    if xu.shape[1] != x.shape[1]:
        raise ValueError("Dimensionality of inducing inputs must match training inputs")
    cov, hc = sc.cov, hyp["cov"]
    nu = xu.shape[0]
    Kuu, Ku, diagK = cov(hc, xu), cov(hc, xu, x), cov(hc, x, "diag")
    if "snu" in hyp:  # learnable inducing noise: log standard deviation
        snu2 = torch.exp(2 * hyp["snu"]) * torch.ones(nu, dtype=x.dtype)
    else:  # gpml default: one per mil of the signal standard deviation
        snu2 = 1e-6 * (torch.trace(Kuu) / nu) * torch.ones(nu, dtype=x.dtype)
    Luu = torch.linalg.cholesky(Kuu + torch.diag(snu2), upper=True)  # Kuu+diag(snu2) = Luu'*Luu
    V = _tri_solve(Luu.T, Ku, upper=False)  # V = inv(Luu')*Ku  =>  V'*V = Q
    g = torch.clamp(diagK - (V * V).sum(0), min=0.0)  # g = diag(K) - diag(Q)

    mvm = lambda v: V.T @ (V @ v) + _col(s * g, v)
    P = lambda v: _tri_solve(Luu, V @ v, upper=True)
    Pt = lambda v: V.T @ _tri_solve(Luu.T, v, upper=False)

    def fun(W: Tensor) -> WFuns:  # ldB2_sparse
        z = s * g * W
        if s == 0.0:
            t = g * W  # VFE: t = g*W
        else:  # FITC / SPEP: t = log(g*s*W + 1)/s, 2nd-order Taylor for tiny z
            small = z < 1e-4
            zs = torch.where(small, torch.zeros_like(z), z)
            t = torch.where(small, g * W * (1 - z / 2 + z**2 / 3), torch.log1p(zs) / s)
        dt = 1.0 / (z + 1.0)
        d = W * dt
        Vd = V * d[None, :]
        Lu = torch.linalg.cholesky(torch.eye(nu, dtype=x.dtype) + V @ Vd.T, upper=True)
        LuV = _tri_solve(Lu.T, V, upper=False)  # Lu'*Lu = I + V*diag(d)*V'
        ldB2 = torch.log(torch.diagonal(Lu)).sum() + t.sum() / 2
        md = lambda r: _col(d, r)
        solveKiW = lambda r: md(r) - md(LuV.T @ (LuV @ md(r)))
        L = torch.empty(0, dtype=x.dtype)
        if want_L:  # Sigma - inv(Kuu): solve_chol(Lu*Luu, I) - solve_chol(Luu, I)
            L = torch.cholesky_inverse(Lu @ Luu, upper=True) - torch.cholesky_inverse(
                Luu, upper=True
            )
        return WFuns(ldB2, solveKiW, L)

    return Approx(mvm, P, Pt, fun, sparse=True, extra={"V": V, "g": g, "Luu": Luu, "xu": xu})
