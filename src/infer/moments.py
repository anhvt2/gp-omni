"""Predictive moments from fitted Gaussian-process posterior state."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.build.covariance import covariance
from gp_omni.build.likelihood import likelihood
from gp_omni.build.mean import mean_function
from gp_omni.build.sparse_covariance import get_xu, sparse_covariance
from gp_omni.infer.posterior import posterior_state


def predict_gp(
    hyp: dict,
    post: posterior_state,
    mean: mean_function,
    cov: covariance | sparse_covariance,
    lik: likelihood,
    x: Tensor,
    xs: Tensor,
    ys: Tensor | None = None,
    full_cov: bool = False,
) -> dict[str, Tensor]:
    """``[ymu, ys2, fmu, fs2, lp] = gp(hyp, inf, mean, cov, lik, x, y, xs, ys)``.

    ``full_cov=True`` also returns ``fcov``, the latent joint covariance over ``xs``
    (exact GPs only).
    """
    if full_cov and isinstance(cov, sparse_covariance):
        raise NotImplementedError("full_cov is not implemented yet for sparse GPs")
    with torch.no_grad():
        hd = {k: v.detach() for k, v in hyp.items()}
        base = cov.cov if isinstance(cov, sparse_covariance) else cov
        kss = base(hd["cov"], xs, "diag")
        if isinstance(cov, sparse_covariance):
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
        fcov = None
        if full_cov:
            kss_full = base(hd["cov"], xs, xs)
            if post.L_kind == "chol":
                fcov = kss_full - V.T @ V
            elif post.L_kind == "matrix":
                fcov = kss_full + Ks.T @ post.L @ Ks
            else:
                fcov = kss_full + Ks.T @ post.L(Ks)
            fcov = (fcov + fcov.T) / 2
        lp, ymu, ys2 = lik.predict(hd["lik"], ys, fmu, fs2, xs)
    out = {"ymu": ymu, "ys2": ys2, "fmu": fmu, "fs2": fs2, "lp": lp}
    if fcov is not None:
        out["fcov"] = fcov
    return out
