"""Hyperparameter training, the role of gpml's ``minimize(hyp, @gp, -n, ...)``."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor

from gp_omni.build.covariance import covariance
from gp_omni.build.likelihood import likelihood
from gp_omni.build.mean import mean_function
from gp_omni.build.sparse_covariance import sparse_covariance
from gp_omni.settings import load_defaults
from gp_omni.train.objectives import inference_method, negative_log_evidence

DEFAULTS = load_defaults("training")

_BIG = 1e12


def optimize(
    hyp: dict[str, Tensor],
    inf: inference_method,
    mean: mean_function,
    cov: covariance | sparse_covariance,
    lik: likelihood,
    x: Tensor,
    y: Tensor,
    opt: dict | None = None,
    max_iter: int = DEFAULTS["sparse_optimizer"]["max_iter"],
    fixed: tuple[str, ...] = tuple(DEFAULTS["sparse_optimizer"]["fixed"]),
    tol: float = DEFAULTS["sparse_optimizer"]["tol"],
) -> tuple[dict[str, Tensor], float]:
    """Minimise the negative log marginal likelihood over all entries of ``hyp`` (including
    ``xu`` and ``snu`` when present) except those named in ``fixed``.

    Returns the optimised ``hyp`` (fresh leaf tensors) and the final ``nlZ``.
    """
    keys = [k for k in hyp if k not in fixed]
    shapes = {k: hyp[k].shape for k in hyp}
    sizes = {k: hyp[k].numel() for k in keys}
    base = {k: hyp[k].detach().clone() for k in hyp}

    def unpack(v: np.ndarray) -> dict[str, Tensor]:
        out, i = {}, 0
        for k in hyp:
            if k in sizes:
                t = torch.tensor(v[i : i + sizes[k]], dtype=torch.float64).reshape(shapes[k])
                out[k] = t.requires_grad_(True)
                i += sizes[k]
            else:
                out[k] = base[k]
        return out

    def fun(v: np.ndarray) -> tuple[float, np.ndarray]:
        h = unpack(v)
        try:
            nlZ, d, _ = negative_log_evidence(h, inf, mean, cov, lik, x, y, opt)
            g = np.concatenate([d[k].reshape(-1).numpy() for k in keys])
            val = float(nlZ.detach())
        except (torch.linalg.LinAlgError, RuntimeError):
            return _BIG, np.zeros(sum(sizes.values()))
        if not np.isfinite(val) or not np.all(np.isfinite(g)):
            return _BIG, np.zeros_like(g)
        return val, g

    v0 = np.concatenate([base[k].reshape(-1).numpy() for k in keys])
    res = minimize(
        fun,
        v0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": DEFAULTS["sparse_optimizer"]["gtol"]},
    )
    out = unpack(res.x)
    return {k: v.detach().clone().requires_grad_(k in sizes) for k, v in out.items()}, float(
        res.fun
    )
