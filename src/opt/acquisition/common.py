"""Helpers shared by every acquisition family: model queries and constraint penalties.

The inner search minimises ``-acquisition(x) * known_constraint(x) * unknown_constraint(x)``:

* ``known_constraint``: 1 when ``x`` satisfies the analytic constraints, 0 when it violates
  one (the objective returns 0 immediately in that case).
* ``unknown_constraint``: ``P(feasible | x)`` from the constraint classifier (``GPC``),
  clipped to ``[0, 1]``; 1 when no classifier could be fitted (e.g. every simulation so far
  was feasible).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from gp_omni.build.gpc import GPC
from gp_omni.build.gpr import GPR
from gp_omni.opt.config import check_constraint, known_constraint_fn
from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("bo")

__all__ = [
    "check_constraint",
    "check_multiple_design_sites",
    "known_constraint_fn",
    "negative_classifier_mse",
    "negative_mse",
    "predict_feasibility",
    "predict_objective",
]


def check_multiple_design_sites(
    x_next: np.ndarray, x_obs: np.ndarray, tol: float = DEFAULTS["duplicate_detection"]["tol"]
) -> bool:
    """True if ``x_next`` duplicates a row of ``x_obs``."""
    x_next = np.asarray(x_next, dtype=float).reshape(1, -1)
    x_obs = np.atleast_2d(x_obs)
    if x_obs.shape[1] != x_next.shape[1]:
        x_obs = x_obs.T
    return bool(np.any(np.linalg.norm(x_obs - x_next, axis=1) < tol))


def _row(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=float).reshape(1, -1))


def predict_objective(model: GPR, x: np.ndarray) -> tuple[float, float]:
    """``[y, s2] = gp(hyp, @infGaussLik, ...)``: predictive mean and variance at one point."""
    out = model.predict(_row(x))
    return float(out["ymu"][0]), max(float(out["ys2"][0]), 0.0)


def predict_feasibility(classifier: GPC | None, x: np.ndarray) -> float:
    """``P(feasible | x)`` in [0, 1]; 1 when there is no usable classifier.

    ``GPC.predict_proba`` is already a probability, so it is used as is (the clip only guards
    against floating-point round-off).
    """
    if classifier is None:
        return 1.0
    try:
        p = float(classifier.predict_proba(_row(x))[0])
    except Exception:  # noqa: BLE001 - a failed classifier must not stop the search
        return 1.0
    return min(max(p, 0.0), 1.0)


def negative_mse(x: np.ndarray, model: GPR) -> float:
    """``-sqrt(s2)`` of the objective model."""
    _, s2 = predict_objective(model, x)
    return -math.sqrt(s2)


def negative_classifier_mse(x: np.ndarray, classifier: GPC) -> float:
    """``-sqrt(s2)`` of the constraint classifier, with ``s2`` the *latent* predictive variance."""
    out = classifier.predict(_row(x))
    return -math.sqrt(max(float(out["fs2"][0]), 0.0))
