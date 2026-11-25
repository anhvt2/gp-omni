"""Acquisitions that need only the marginal posterior ``N(y, s2)`` at the candidate (maximisation).

Every function maps the posterior at one point to a score; larger is better.

======  ==========================================================  =======================
name    acquisition                                                 reference
======  ==========================================================  =======================
``PI``  ``Phi(gamma)``                                              Kushner (1964)
``EI``  ``s (gamma Phi(gamma) + phi(gamma))``                       Mockus et al. (1978)
``UCB`` ``y + sqrt(kappa) s``, ``kappa`` from the regret bound      Srinivas et al. (2010)
``PE``  ``s2`` (pure exploration = maximum differential entropy)    Contal et al. (2013)
``MES`` mean over ``y*`` samples of the entropy reduction           Wang & Jegelka (2017)
======  ==========================================================  =======================

with ``gamma = (y - f_best - xi) / s``. ``PI`` and ``EI`` take an optional exploration
offset ``xi >= 0``. ``MES`` needs samples of the optimum value ``y*``; build them once per
proposal with :func:`sample_max_values`.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm

from gp_omni.build.gpr import GPR
from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("bo")


def ucb_kappa(t: int, d: int, delta: float = DEFAULTS["acquisition"]["delta"]) -> float:
    """``kappa`` of option 2 in Srinivas et al., theorem 2 (as used by Daniel et al.)."""
    return 2 * math.log(t ** (d / 2 + 2) * math.pi**2 / 3 / delta)


def probability_of_improvement(y: float, s2: float, f_best: float, xi: float = 0.0) -> float:
    s = math.sqrt(s2)
    if s == 0.0:  # PI of a deterministic point is a step function
        return float(y - f_best - xi > 0)
    return float(norm.cdf((y - f_best - xi) / s))


def expected_improvement(y: float, s2: float, f_best: float, xi: float = 0.0) -> float:
    s = math.sqrt(s2)
    if s == 0.0:
        return max(y - f_best - xi, 0.0)
    gamma = (y - f_best - xi) / s
    return s * (gamma * norm.cdf(gamma) + norm.pdf(gamma))


def upper_confidence_bound(y: float, s2: float, kappa: float) -> float:
    return y + math.sqrt(kappa) * math.sqrt(s2)


def pure_exploration(s2: float) -> float:
    """Posterior variance; maximising it maximises the differential entropy of ``f(x)``."""
    return s2


def max_value_entropy(y: float, s2: float, y_star: np.ndarray) -> float:
    """MES: ``mean_k [gamma_k phi(gamma_k) / (2 Phi(gamma_k)) - log Phi(gamma_k)]``.

    ``gamma_k = (y*_k - y) / s``; ``y_star`` holds samples of the optimum value.
    """
    s = math.sqrt(s2)
    if s == 0.0:
        return 0.0
    gamma = (np.asarray(y_star, dtype=float) - y) / s
    cdf = np.clip(norm.cdf(gamma), 1e-300, None)
    return float(np.mean(gamma * norm.pdf(gamma) / (2 * cdf) - np.log(cdf)))


def sample_max_values(
    model: GPR,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    f_best: float,
    rng: np.random.Generator,
    n_samples: int = 10,
    n_candidates: int = 1000,
) -> np.ndarray:
    """Draw ``y*`` from the Gumbel approximation of the posterior maximum (Wang & Jegelka).

    ``P(max f <= y)`` is approximated by ``prod_i Phi((y - mu_i) / s_i)`` over random
    candidates; its quartiles fit a Gumbel distribution that is then sampled. Samples are
    floored at ``f_best`` because the optimum cannot be below the best observation.
    """
    x_lb, x_ub = np.asarray(x_lb, float), np.asarray(x_ub, float)
    cand = x_lb + rng.random((n_candidates, len(x_lb))) * (x_ub - x_lb)
    out = model.predict(torch.as_tensor(cand))
    mu = np.asarray(out["ymu"], dtype=float).reshape(-1)
    sd = np.sqrt(np.clip(np.asarray(out["ys2"], dtype=float).reshape(-1), 1e-18, None))

    def quantile(p: float) -> float:
        def gap(y):
            return float(np.sum(norm.logcdf((y - mu) / sd)) - math.log(p))

        lo, hi = float(np.max(mu)), float(np.max(mu + 5 * sd))
        while gap(hi) < 0:  # widen until the bracket contains the root
            hi += float(np.max(sd))
        return brentq(gap, lo - 5 * float(np.max(sd)), hi)

    q25, q50, q75 = quantile(0.25), quantile(0.5), quantile(0.75)
    scale = max((q25 - q75) / (math.log(math.log(4 / 3)) - math.log(math.log(4))), 1e-12)
    r = rng.random(n_samples)
    return np.maximum(q50 - scale * np.log(-np.log(r)), f_best)


def pointwise_value(
    name: str,
    y: float,
    s2: float,
    f_best: float,
    t: int,
    d: int,
    delta: float = DEFAULTS["acquisition"]["delta"],
    xi: float = 0.0,
    y_star: np.ndarray | None = None,
) -> float:
    """Score ``N(y, s2)`` with PI, EI, UCB, PE or MES (``y_star`` required for MES)."""
    if name == "UCB":
        return upper_confidence_bound(y, s2, ucb_kappa(t, d, delta))
    if name == "PI":
        return probability_of_improvement(y, s2, f_best, xi)
    if name == "EI":
        return expected_improvement(y, s2, f_best, xi)
    if name == "PE":
        return pure_exploration(s2)
    if name == "MES":
        if y_star is None:
            raise ValueError("MES needs context=prepare_context('MES', ...)")
        return max_value_entropy(y, s2, y_star)
    raise ValueError(f"not a pointwise acquisition: {name!r}")
