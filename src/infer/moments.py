"""Predictive moments from fitted Gaussian-process posterior state."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.build.covariance import Cov
from gp_omni.build.likelihood import Lik
from gp_omni.build.mean import Mean
from gp_omni.build.sparse_covariance import SparseCov, get_xu
from gp_omni.infer.posterior import Posterior


def predict_gp(
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
