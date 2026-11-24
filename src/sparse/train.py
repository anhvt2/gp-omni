"""Hyperparameter training, the role of gpml's ``minimize(hyp, @gp, -n, ...)``."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor

from gp_omni.settings import load_defaults
from gp_omni.sparse.apx import SparseCov
from gp_omni.sparse.cov import Cov
from gp_omni.sparse.gp import Inference, gp_nlz
from gp_omni.sparse.lik import Lik
from gp_omni.sparse.mean import Mean

DEFAULTS = load_defaults("training")

_BIG = 1e12


def train(
    hyp: dict[str, Tensor],
    inf: Inference,
    mean: Mean,
    cov: Cov | SparseCov,
    lik: Lik,
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
            nlZ, d, _ = gp_nlz(h, inf, mean, cov, lik, x, y, opt)
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
