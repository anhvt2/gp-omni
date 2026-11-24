"""Penalised acquisition functions.

The objective of the CMA-ES inner search is (minimised)::

    -acquisition(x) * known_constraint(x) * unknown_constraint(x)

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
from scipy.stats import norm

from gp_omni.build.gpc import GPC
from gp_omni.build.gpr import GPR
from gp_omni.settings import load_defaults
from gp_omni.train.bo_config import check_constraint, known_constraint_fn

DEFAULTS = load_defaults("bo")


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


def acquisition_value(
    name: str,
    y: float,
    s2: float,
    f_best: float,
    t: int,
    d: int,
    delta: float = DEFAULTS["acquisition"]["delta"],
) -> float:
    """PI / EI / UCB for a maximisation problem."""
    s = math.sqrt(s2)
    if name == "UCB":
        # option-2, Srinivas et al. theorem 2 / Daniel et al.
        kappa = 2 * math.log(t ** (d / 2 + 2) * math.pi**2 / 3 / delta)
        return y + math.sqrt(kappa) * s
    if s == 0.0:  # avoid dividing by zero; PI/EI of a deterministic point are step functions
        gain = y - f_best
        return float(gain > 0) if name == "PI" else max(gain, 0.0)
    gamma = (y - f_best) / s
    if name == "PI":
        return float(norm.cdf(gamma))
    if name == "EI":
        return s * (gamma * norm.cdf(gamma) + norm.pdf(gamma))
    raise ValueError(f"acquisition function must be PI, EI or UCB, got {name!r}")


def negative_acquisition(
    x: np.ndarray,
    model: GPR,
    classifier: GPC | None,
    acquisition: str,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    known_constraint: known_constraint_fn = check_constraint,
) -> float:
    """``-acquisition * known * unknown``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    known = known_constraint(x)
    if known == 0:
        return 0.0
    unknown = predict_feasibility(classifier, x)
    y, s2 = predict_objective(model, x)
    t, d = x_obs.shape
    a = acquisition_value(acquisition, y, s2, float(np.max(y_obs)), t, d)
    return -a * known * unknown


def negative_mse(x: np.ndarray, model: GPR) -> float:
    """``-sqrt(s2)`` of the objective model."""
    _, s2 = predict_objective(model, x)
    return -math.sqrt(s2)


def negative_classifier_mse(x: np.ndarray, classifier: GPC) -> float:
    """``-sqrt(s2)`` of the constraint classifier, with ``s2`` the *latent* predictive variance."""
    out = classifier.predict(_row(x))
    return -math.sqrt(max(float(out["fs2"][0]), 0.0))
