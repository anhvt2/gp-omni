"""Penalised acquisition functions: port of ``bayesOptSrc-gpml/calculateNegative*.m``.

The objective of the CMA-ES inner search is (minimised)::

    -acquisition(x) * known_constraint(x) * unknown_constraint(x)

* ``known_constraint``: ``checkConstraint.m``; 1 when ``x`` satisfies the analytic constraints,
  0 when it violates one (the MATLAB code returns 0 immediately in that case).
* ``unknown_constraint``: ``P(feasible | x)`` from the constraint classifier (``OmniGPC``),
  clipped to ``[0, 1]``; 1 when no classifier could be fitted (e.g. every simulation so far
  was feasible), as in the MATLAB ``try/catch``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from scipy.stats import norm

from gp_omni.sparse.omni import OmniGP, OmniGPC

KnownConstraint = Callable[[np.ndarray], float]


def check_constraint(x: np.ndarray) -> float:
    """``checkConstraint.m``: 1 if no known constraint is violated, 0 otherwise.

    The MATLAB default accepts everything; pass your own callable to ``AsyncBO`` to override.
    """
    return 1.0


def check_multiple_design_sites(x_next: np.ndarray, x_obs: np.ndarray, tol: float = 1e-4) -> bool:
    """``checkMultipleDesignSites.m``: True if ``x_next`` duplicates a row of ``x_obs``."""
    x_next = np.asarray(x_next, dtype=float).reshape(1, -1)
    x_obs = np.atleast_2d(x_obs)
    if x_obs.shape[1] != x_next.shape[1]:
        x_obs = x_obs.T
    return bool(np.any(np.linalg.norm(x_obs - x_next, axis=1) < tol))


def _row(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=float).reshape(1, -1))


def predict_objective(model: OmniGP, x: np.ndarray) -> tuple[float, float]:
    """``[y, s2] = gp(hyp, @infGaussLik, ...)``: predictive mean and variance at one point."""
    out = model.predict(_row(x))
    return float(out["ymu"][0]), max(float(out["ys2"][0]), 0.0)


def predict_feasibility(
    classifier: OmniGPC | None, x: np.ndarray, relax_alpha: float | None = None
) -> float:
    """``P(feasible | x)`` clipped to [0, 1]; 1 when there is no usable classifier.

    With ``relax_alpha`` (``classifPredictor.m``, 0.9 there) probabilities above ``alpha`` are
    rounded to 1 and below ``1 - alpha`` to 0, so only the uncertain band is penalised softly.
    """
    if classifier is None:
        return 1.0
    try:
        p = float(classifier.predict_proba(_row(x))[0])
    except Exception:  # noqa: BLE001 - mirrors the MATLAB try/catch safeguard
        return 1.0
    p = min(max(p, 0.0), 1.0)
    if relax_alpha is not None:
        p = 1.0 if p > relax_alpha else 0.0 if p < 1 - relax_alpha else p
    return p


def acquisition_value(
    name: str, y: float, s2: float, f_best: float, t: int, d: int, delta: float = 0.95
) -> float:
    """PI / EI / UCB for a maximisation problem (``calculateNegativeAcquisitionFunction.m``)."""
    s = math.sqrt(s2)
    if name == "UCB":
        # option-2, Srinivas et al. theorem 2 / Daniel et al.
        kappa = 2 * math.log(t ** (d / 2 + 2) * math.pi**2 / 3 / delta)
        return y + math.sqrt(kappa) * s
    if s == 0.0:  # MATLAB divides by zero here; PI/EI of a deterministic point are step functions
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
    model: OmniGP,
    classifier: OmniGPC | None,
    acquisition: str,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    known_constraint: KnownConstraint = check_constraint,
    relax_alpha: float | None = None,
) -> float:
    """``calculateNegativeAcquisitionFunction.m``: ``-acquisition * known * unknown``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    known = known_constraint(x)
    if known == 0:
        return 0.0
    unknown = predict_feasibility(classifier, x, relax_alpha)
    y, s2 = predict_objective(model, x)
    t, d = x_obs.shape
    a = acquisition_value(acquisition, y, s2, float(np.max(y_obs)), t, d)
    return -a * known * unknown


def negative_mse(x: np.ndarray, model: OmniGP) -> float:
    """``calculateNegativeMSE.m``: ``-sqrt(s2)`` of the objective model."""
    _, s2 = predict_objective(model, x)
    return -math.sqrt(s2)


def negative_classifier_mse(x: np.ndarray, classifier: OmniGPC) -> float:
    """``calculateNegativeClfMSE.m``: ``-sqrt(s2)`` of the constraint classifier.

    The MATLAB code rescales the latent ``y`` to a probability but takes the sqrt of the
    *latent* variance ``s2``; the same quantity (the latent predictive variance) is used here.
    """
    out = classifier.predict(_row(x))
    return -math.sqrt(max(float(out["fs2"][0]), 0.0))
