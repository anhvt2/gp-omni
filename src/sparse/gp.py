"""gpml's ``gp.m``: marginal likelihood with derivatives, and prediction."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from gp_omni.sparse.apx import SparseCov, get_xu
from gp_omni.sparse.cov import Cov
from gp_omni.sparse.inf import Posterior, inf_gauss_lik
from gp_omni.sparse.lik import Lik
from gp_omni.sparse.mean import Mean

Inference = Callable[..., tuple[Posterior, Tensor]]


def make_hyp(**kw) -> dict[str, Tensor]:
    """Build ``hyp`` (gpml's struct): float64 tensors for ``mean``, ``cov``, ``lik`` and
    optional ``xu`` (nu x D) / ``snu``."""
    return {k: _t(k, v) for k, v in kw.items()}


def _t(key: str, v) -> Tensor:
    t = torch.as_tensor(v, dtype=torch.float64)
    return t if key == "xu" else t.reshape(-1)


def check_hyp(hyp: dict, mean: Mean, cov: Cov | SparseCov, lik: Lik, dim: int) -> None:
    """gpml's hyperparameter-count checks."""
    base = cov.cov if isinstance(cov, SparseCov) else cov
    for key, need in (
        ("mean", mean.num_hyp(dim)),
        ("cov", base.num_hyp(dim)),
        ("lik", lik.num_hyp()),
    ):
        if hyp.get(key, torch.empty(0)).numel() != need:
            raise ValueError(f"Number of {key} hyperparameters disagree with the {key} function")


def gp_nlz(
    hyp: dict,
    inf: Inference,
    mean: Mean,
    cov: Cov | SparseCov,
    lik: Lik,
    x: Tensor,
    y: Tensor,
    opt: dict | None = None,
    grad: bool = True,
) -> tuple[Tensor, dict[str, Tensor] | None, Posterior]:
    """``[nlZ, dnlZ, post] = gp(hyp, inf, mean, cov, lik, x, y)``.

    ``dnlZ`` has the same keys as ``hyp`` (all tensors in ``hyp`` that require grad are
    differentiated; others are treated as constants)."""
    check_hyp(hyp, mean, cov, lik, x.shape[1])
    post, nlZ = inf(hyp, mean, cov, lik, x, y, opt)
    dnlZ = None
    if grad:
        keys = [k for k, v in hyp.items() if v.requires_grad]
        gr = torch.autograd.grad(nlZ, [hyp[k] for k in keys], allow_unused=True)
        dnlZ = {k: (torch.zeros_like(hyp[k]) if g is None else g) for k, g in zip(keys, gr)}
    return nlZ, dnlZ, post


def gp_predict(
    hyp: dict,
    post: Posterior,
    mean: Mean,
    cov: Cov | SparseCov,
    lik: Lik,
    x: Tensor,
    xs: Tensor,
    ys: Tensor | None = None,
) -> dict[str, Tensor]:
    """``[ymu, ys2, fmu, fs2, lp] = gp(hyp, inf, mean, cov, lik, x, y, xs, ys)``."""
    with torch.no_grad():
        hd = {k: v.detach() for k, v in hyp.items()}
        base = cov.cov if isinstance(cov, SparseCov) else cov
        kss = base(hd["cov"], xs, "diag")
        if isinstance(cov, SparseCov):
            xu = get_xu(hd, cov)
            Ks = base(hd["cov"], xu, xs)  # cross-covariances to the inducing inputs
        else:
            Ks = base(hd["cov"], x, xs)
        fmu = mean(hd["mean"], xs) + Ks.T @ post.alpha
        if post.L_kind == "chol":
            V = torch.linalg.solve_triangular(post.L.T, post.sW[:, None] * Ks, upper=False)
            fs2 = kss - (V * V).sum(0)
        elif post.L_kind == "matrix":
            fs2 = kss + (Ks * (post.L @ Ks)).sum(0)
        else:
            fs2 = kss + (Ks * post.L(Ks)).sum(0)
        fs2 = torch.clamp(fs2, min=0.0)
        lp, ymu, ys2 = lik.predict(hd["lik"], ys, fmu, fs2, xs)
    return {"ymu": ymu, "ys2": ys2, "fmu": fmu, "fs2": fs2, "lp": lp}


__all__ = ["check_hyp", "gp_nlz", "gp_predict", "inf_gauss_lik", "make_hyp"]
