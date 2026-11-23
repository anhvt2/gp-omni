"""Inference methods that work with exact and sparse covariance approximations.

Ports of gpml's ``infGaussLik.m`` and ``infLaplace.m`` (v4.2). Each returns ``(post, nlZ)``
with gpml's posterior parameters ``alpha``, ``sW`` and ``L``. Derivatives of ``nlZ`` are not
hand-coded as in gpml: they come from autograd through ``nlZ`` (see ``gp.gp_nlz``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from gp_omni.sparse.apx import Approx, SparseCov, apx
from gp_omni.sparse.cov import Cov
from gp_omni.sparse.lik import Lik, LikGauss
from gp_omni.sparse.mean import Mean


@dataclass
class Posterior:
    """gpml's ``post`` struct. ``L`` is an upper Cholesky factor (``L_kind='chol'``), a
    matrix (``'matrix'``) or a callable acting on cross-covariances (``'callable'``)."""

    alpha: Tensor
    sW: Tensor
    L: Tensor | Callable[[Tensor], Tensor]
    L_kind: str


def inf_gauss_lik(
    hyp: dict,
    mean: Mean,
    cov: Cov | SparseCov,
    lik: Lik,
    x: Tensor,
    y: Tensor,
    opt: dict | None = None,
) -> tuple[Posterior, Tensor]:
    """Exact inference for Gaussian likelihoods; sparse if ``cov`` is a ``SparseCov``."""
    if not isinstance(lik, LikGauss):
        raise TypeError("Exact inference only possible with Gaussian likelihood")
    m = mean(hyp["mean"], x)
    sn2 = lik.noise_var(hyp["lik"], x)  # per-sample (a constant for likGauss)
    W = 1.0 / sn2  # noise precision
    K = apx(hyp, cov, x, opt)
    wf = K.fun(W)
    alpha = wf.solveKiW(y - m)
    post = Posterior(K.P(alpha), torch.sqrt(W), wf.L, "matrix" if K.sparse else "chol")
    nlZ = (y - m) @ alpha / 2 + wf.ldB2 + torch.log(2 * math.pi * sn2).sum() / 2
    return post, nlZ


def _psi(alpha: Tensor, m: Tensor, mvm, likfun):
    f = mvm(alpha) + m
    lp, dlp, d2lp = likfun(f)
    return (alpha @ (f - m)) / 2 - lp.sum(), f, dlp, -d2lp


def _irls(alpha: Tensor, m: Tensor, K: Approx, likfun, opt: dict) -> Tensor:
    """Newton iterations with step-size search, as gpml's ``irls`` (run without autograd)."""
    maxit = int(opt.get("irls_maxit", 20))
    wmin = float(opt.get("irls_Wmin", 0.0))
    tol = float(opt.get("irls_tol", 1e-6))
    psi_new, f, dlp, W = _psi(alpha, m, K.mvm, likfun)
    psi_old, it = math.inf, 0
    while psi_old - psi_new > tol and it < maxit:
        psi_old, it = psi_new, it + 1
        W = torch.clamp(W, min=wmin)
        wf = K.fun(W)
        b = W * (f - m) + dlp
        dalpha = b - wf.solveKiW(K.mvm(b)) - alpha  # Newton direction
        best = (psi_new, alpha, f, dlp, W)
        for s in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0):  # backtracking line search
            cand = alpha + s * dalpha
            p, fc, dc, Wc = _psi(cand, m, K.mvm, likfun)
            if p <= psi_old:
                best = (p, cand, fc, dc, Wc)
                break
        psi_new, alpha, f, dlp, W = best
    return alpha


def inf_laplace(
    hyp: dict,
    mean: Mean,
    cov: Cov | SparseCov,
    lik: Lik,
    x: Tensor,
    y: Tensor,
    opt: dict | None = None,
) -> tuple[Posterior, Tensor]:
    """Laplace approximation; sparse if ``cov`` is a ``SparseCov``.

    The posterior mode is found without autograd (gpml's ``irls``). The returned ``nlZ`` is
    then rebuilt with one differentiable Newton step from the mode, which (by the implicit
    function theorem) gives exactly gpml's total derivative including the ``dfhat`` term.
    """
    opt = opt or {}
    n = x.shape[0]
    hd = {k: v.detach() for k, v in hyp.items()}
    with torch.no_grad():
        K0 = apx(hd, cov, x, opt, want_L=False)
        m0 = mean(hd["mean"], x)
        lf0 = lambda f: tuple(t.detach() for t in lik.laplace(hd["lik"], y, f))
        alpha0 = _irls(torch.zeros(n, dtype=x.dtype), m0, K0, lf0, opt)

    K = apx(hyp, cov, x, opt)
    m = mean(hyp["mean"], x)
    likfun = lambda f: lik.laplace(hyp["lik"], y, f)
    f0 = K.mvm(alpha0) + m
    _, dlp0, d2lp0 = likfun(f0)
    W0 = -d2lp0
    b = W0 * (f0 - m) + dlp0
    alpha = b - K.fun(W0).solveKiW(K.mvm(b))  # differentiable Newton step at the mode
    f = K.mvm(alpha) + m
    lp, _, d2lp = likfun(f)
    W = -d2lp
    wf = K.fun(W)
    nlZ = alpha @ (f - m) / 2 - lp.sum() + wf.ldB2
    sW = torch.sqrt(torch.abs(W)) * torch.sign(W)
    post = Posterior(K.P(alpha), sW, lambda r: -K.P(wf.solveKiW(K.Pt(r))), "callable")
    return post, nlZ
