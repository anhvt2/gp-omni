"""Optimize sparse-GP hyperparameters and inducing locations."""

from __future__ import annotations

import torch

from gp_omni.build.inducing import init_inducing
from gp_omni.build.likelihood import gaussian
from gp_omni.settings import load_defaults
from gp_omni.train.inducing import vfe_xu_opt
from gp_omni.train.objectives import negative_log_evidence
from gp_omni.train.optimize import optimize

DEFAULTS = load_defaults("models")


def fit(
    model,
    x,
    y,
    learn_xu=DEFAULTS["sparse_fit"]["learn_xu"],
    swap=DEFAULTS["sparse_fit"]["swap"],
    max_iter=DEFAULTS["sparse_fit"]["max_iter"],
    hyp=None,
):
    """Minimize sparse negative log evidence, optionally swapping inducing inputs."""
    x, y = x.to(torch.float64), y.to(torch.float64).reshape(-1)
    if y.shape[0] != x.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if not isinstance(model.lik, gaussian) and not set(y.unique().tolist()) <= {-1.0, 1.0}:
        raise ValueError("classification labels must be in {-1, +1}")
    h = hyp or model._initial_hyp(x, y, learn_xu)
    if hyp is not None:
        model._xu0 = h.get("xu", init_inducing(x, model.nu, model.init, model.seed))
    opt = {"s": model.s}
    if swap > 0:
        if not (isinstance(model.lik, gaussian) and model.s == 0.0):
            raise ValueError("vfe_xu_opt swaps need s=0 and a Gaussian likelihood")
        h["xu"] = model._xu0.clone()
        h, _ = vfe_xu_opt(h, model.mean, model.cov, x, y, x, swap)
        model._xu0 = h["xu"].detach()
        if not learn_xu:
            h.pop("xu")
    h, nlZ = optimize(h, model.inf, model.mean, model._cov(), model.lik, x, y, opt, max_iter)
    model.hyp, model.nlZ, model._x, model._y, model._opt = h, nlZ, x, y, opt
    _, _, model.post = negative_log_evidence(
        h, model.inf, model.mean, model._cov(), model.lik, x, y, opt, grad=False
    )
    return model
