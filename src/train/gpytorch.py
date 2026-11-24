"""Training helpers around BoTorch's ``fit_gpytorch_mll``."""

from __future__ import annotations

import copy
import warnings

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.model import Model
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO

from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("training")


def _fit(
    model,
    mll,
    steps: int = DEFAULTS["exact_optimizer"]["steps"],
    lr: float = DEFAULTS["exact_optimizer"]["lr"],
) -> None:
    """BoTorch's scipy fitter for BoTorch models; plain Adam for other GPyTorch models."""
    if isinstance(model, Model):
        fit_gpytorch_mll(mll)
        return
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = -mll(model(*model.train_inputs), model.train_targets).sum()
        loss.backward()
        opt.step()


def _mll_value(model, mll) -> float:
    model.train()
    with torch.no_grad():
        out = model(*model.train_inputs)
        val = mll(out, model.train_targets)
    return float(val.sum())


def fit_exact(
    model,
    num_restarts: int = DEFAULTS["exact"]["num_restarts"],
    noise: float = DEFAULTS["exact"]["noise"],
    seed: int = DEFAULTS["exact"]["seed"],
):
    """Maximise the exact marginal likelihood; optionally with random restarts.

    Restart ``k`` perturbs the (unconstrained) hyperparameters of the original state with
    ``N(0, noise^2)`` noise, re-fits, and the best marginal likelihood wins.
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    init = copy.deepcopy(model.state_dict())
    _fit(model, mll)
    best, best_state = _mll_value(model, mll), copy.deepcopy(model.state_dict())
    gen = torch.Generator().manual_seed(seed)
    for _ in range(num_restarts):
        model.load_state_dict(init)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.startswith("raw_") or ".raw_" in name:
                    p.add_(noise * torch.randn(p.shape, generator=gen, dtype=p.dtype))
        try:
            _fit(model, mll)
        except Exception as err:  # noqa: BLE001 - a bad restart must not abort the others
            warnings.warn(f"fit_exact: restart failed and was skipped ({err!r})", stacklevel=2)
            continue
        val = _mll_value(model, mll)
        if val > best:
            best, best_state = val, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model


def fit_variational(
    model,
    likelihood,
    train_x,
    train_y,
    epochs: int = DEFAULTS["variational"]["epochs"],
    lr: float = DEFAULTS["variational"]["lr"],
) -> float:
    """Adam on the ELBO for an ``ApproximateGP`` (svgp, classification). Returns final loss."""
    elbo = VariationalELBO(likelihood, model, num_data=train_y.shape[0])
    params = list(model.parameters()) + list(likelihood.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    model.train()
    likelihood.train()
    loss = torch.zeros(())
    for _ in range(epochs):
        opt.zero_grad()
        loss = -elbo(model(train_x), train_y)
        loss.backward()
        opt.step()
    model.eval()
    likelihood.eval()
    return float(loss.detach())
