"""Hallucination: what the models see for points whose true result is not available.

Parallel BO never changes the acquisition function. It only changes the data the models
are fitted on: infeasible and pending points get the posterior mean as their output
(kriging believer), after which the ordinary sequential acquisition proposes the next point.
"""

from __future__ import annotations

import numpy as np
import torch

from gp_omni.build.gpc import GPC
from gp_omni.build.gpr import GPR
from gp_omni.opt.acquisition import predict_objective


def fit_models(
    x_obs: np.ndarray, y_obs: np.ndarray, f_obs: np.ndarray, max_iter: int
) -> tuple[GPR | None, GPC | None, np.ndarray | None]:
    """Hallucinate infeasible / pending outputs, then fit objective model and classifier.

    1. fit the objective on feasible points only; 2. replace the outputs of infeasible
    and pending points with that model's prediction; 3. refit on everything;
    4. fit the constraint classifier on ``f_obs`` (skipped while every label is the same).

    Returns ``(model, classifier, y_hallucinated)``; ``model`` is ``None`` with fewer than
    two feasible points.
    """
    x_train = torch.as_tensor(x_obs)
    feasible = torch.as_tensor(f_obs > 0)
    model = classifier = y_hallucinated = None
    if feasible.sum() >= 2:
        interp = GPR().fit(x_train[feasible], torch.as_tensor(y_obs)[feasible], max_iter=max_iter)
        y_obs = y_obs.copy()
        for i in np.flatnonzero(f_obs == 0):
            y_obs[i] = predict_objective(interp, x_obs[i])[0]
        model = GPR().fit(x_train, torch.as_tensor(y_obs), max_iter=max_iter)
        y_hallucinated = y_obs
    if 0 < feasible.sum() < len(f_obs):
        classifier = GPC().fit(x_train, torch.as_tensor(f_obs > 0), max_iter=max_iter)
    return model, classifier, y_hallucinated


def hallucinate_point(
    model: GPR,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    f_obs: np.ndarray,
    x_new: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GPR, GPC | None]:
    """Append ``x_new`` with ``model``'s mean there, labelled feasible, and refit both models."""
    mu, _ = predict_objective(model, x_new)
    x_obs = np.vstack([x_obs, x_new])
    y_obs = np.append(y_obs, mu)
    f_obs = np.append(f_obs, 1.0)
    new_model = GPR().fit(torch.as_tensor(x_obs), torch.as_tensor(y_obs), max_iter=max_iter)
    classifier = None
    if (f_obs == 0).any():
        classifier = GPC().fit(
            torch.as_tensor(x_obs), torch.as_tensor(f_obs > 0), max_iter=max_iter
        )
    return x_obs, y_obs, f_obs, new_model, classifier
