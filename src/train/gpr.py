"""Standardize task targets and optimize Gaussian-process evidence."""

from __future__ import annotations

import math

import torch

from gp_omni.build.hyperparameters import make_hyperparameters
from gp_omni.build.inducing import init_inducing
from gp_omni.build.likelihood import LikGauss
from gp_omni.build.multitask import (
    augment,
)
from gp_omni.settings import load_defaults
from gp_omni.train.objectives import negative_log_evidence
from gp_omni.train.optimize import optimize

DEFAULTS = load_defaults("models")


def fit(
    model,
    x,
    y,
    output=None,
    fidelity=None,
    max_iter=DEFAULTS["gpr_fit"]["max_iter"],
    learn_xu=DEFAULTS["gpr_fit"]["learn_xu"],
):
    """Fit task-standardized evidence and return the trained model."""
    x, y = x.to(torch.float64), y.to(torch.float64).reshape(-1)
    n, D = x.shape
    model._D = D
    if y.shape[0] != n:
        raise ValueError("x and y must have the same number of rows")
    o = torch.zeros(n, dtype=torch.long) if output is None else torch.as_tensor(output).long()
    f = torch.zeros(n, dtype=torch.long) if fidelity is None else torch.as_tensor(fidelity).long()
    if int(o.max()) >= model.O or int(f.max()) >= model.L or int(o.min()) < 0 or int(f.min()) < 0:
        raise ValueError(
            "output / fidelity indices must lie in [0, num_outputs) / [0, num_fidelities)"
        )
    model._build(D)
    gauss = isinstance(model.lik, LikGauss)
    if not gauss and not set(y.unique().tolist()) <= {-1.0, 1.0}:
        raise ValueError("classification labels must be in {-1, +1}")

    # per-task standardisation (Gaussian likelihoods only)
    T = model.O * model.L
    model._mu, model._sd = torch.zeros(T), torch.ones(T)
    t = o * model.L + f
    ys = y.clone()
    if gauss:
        for k in range(T):
            m = t == k
            if m.sum() >= 1:
                model._mu[k] = y[m].mean()
            if m.sum() >= 2 and y[m].std() > 0:
                model._sd[k] = y[m].std()
        ys = (y - model._mu[t]) / model._sd[t]

    sf = DEFAULTS["initialization"][
        "regression_signal_std" if gauss else "classification_signal_std"
    ]
    bh = model._base_hyp(x, sf)
    if model.multitask:
        hyp = make_hyperparameters(
            mean=torch.zeros(T),
            cov=model.cov.initial_hyp(bh),
            lik=torch.full((T,), math.log(DEFAULTS["initialization"]["noise_std"]))
            if gauss
            else torch.zeros(0),
        )
    else:
        hyp = make_hyperparameters(
            mean=torch.zeros(1),
            cov=bh,
            lik=torch.full((1,), math.log(DEFAULTS["initialization"]["noise_std"]))
            if gauss
            else torch.zeros(0),
        )
    xu0 = None
    if model.nu is not None:
        xu0 = init_inducing(x, model.nu, model.init, model.seed)
        if learn_xu:
            hyp["xu"] = xu0.clone()
    model._xu0 = xu0
    model._xa = augment(x, o, f) if model.multitask else x
    model._ys = ys
    cov = model._sparse_cov(xu0)
    opt = {"s": model.s}
    hyp, model.nlZ = optimize(
        hyp, model.inf, model.mean, cov, model.lik, model._xa, ys, opt, max_iter
    )
    model.hyp, model._opt = hyp, opt
    _, _, model.post = negative_log_evidence(
        hyp, model.inf, model.mean, cov, model.lik, model._xa, ys, opt, grad=False
    )
    return model
