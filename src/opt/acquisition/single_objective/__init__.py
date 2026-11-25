"""Acquisition functions for a single objective (maximisation).

======  ===========================================================  =========================
name    acquisition                                                  reference
======  ===========================================================  =========================
``PI``  ``Phi(gamma)``                                               Kushner (1964)
``EI``  ``s (gamma Phi(gamma) + phi(gamma))``                        Mockus et al. (1978)
``UCB`` ``y + sqrt(kappa) s``, ``kappa`` from the regret bound       Srinivas et al. (2010)
``PE``  ``s2`` (pure exploration = maximum differential entropy)     Contal et al. (2013)
``MES`` mean over ``y*`` samples of the entropy reduction            Wang & Jegelka (2017)
======  ===========================================================  =========================

with ``gamma = (y - f_best - xi) / s``. All of them are sequential: parallel runs hallucinate
pending points first (:mod:`gp_omni.opt.parallel`). ``MES`` needs a
per-proposal ``context`` from :func:`prepare_context` (optimum-value samples) that stays
fixed while the inner search runs.
"""

from __future__ import annotations

import numpy as np

from gp_omni.build.gpc import GPC
from gp_omni.build.gpr import GPR
from gp_omni.opt.acquisition.common import (
    check_constraint,
    known_constraint_fn,
    predict_feasibility,
    predict_objective,
)
from gp_omni.opt.acquisition.single_objective.pointwise import (
    DEFAULTS,
    expected_improvement,
    max_value_entropy,
    pointwise_value,
    probability_of_improvement,
    pure_exploration,
    sample_max_values,
    ucb_kappa,
    upper_confidence_bound,
)

#: scored from ``(y, s2)`` alone
POINTWISE = ("PI", "EI", "UCB", "PE")
#: need the joint posterior or samples, prepared once per proposal
CONTEXTUAL = ("MES",)
NAMES = POINTWISE + CONTEXTUAL
RANDOM = "MC"  # uniform random sampling, handled by the search

__all__ = [
    "CONTEXTUAL",
    "NAMES",
    "POINTWISE",
    "RANDOM",
    "acquisition_value",
    "check_acquisition_name",
    "expected_improvement",
    "max_value_entropy",
    "negative_acquisition",
    "prepare_context",
    "probability_of_improvement",
    "pure_exploration",
    "sample_max_values",
    "ucb_kappa",
    "upper_confidence_bound",
]


def check_acquisition_name(name: str) -> None:
    if name not in NAMES and name != RANDOM:
        raise ValueError(
            f"acquisition function must be one of {', '.join(NAMES)} (or {RANDOM!r}), got {name!r}"
        )


def prepare_context(
    name: str,
    model: GPR,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Per-proposal quantities that must stay fixed while the inner search runs."""
    if name == "MES":
        return {"y_star": sample_max_values(model, x_lb, x_ub, float(np.max(y_obs)), rng)}
    return {}


def acquisition_value(
    name: str,
    y: float,
    s2: float,
    f_best: float,
    t: int,
    d: int,
    delta: float = DEFAULTS["acquisition"]["delta"],
    xi: float = 0.0,
    context: dict | None = None,
) -> float:
    """Score the marginal posterior ``N(y, s2)`` (PI, EI, UCB, PE, MES) for maximisation."""
    if name not in NAMES:
        check_acquisition_name(name)
    y_star = None if context is None else context.get("y_star")
    return pointwise_value(name, y, s2, f_best, t, d, delta, xi, y_star)


def negative_acquisition(
    x: np.ndarray,
    model: GPR,
    classifier: GPC | None,
    acquisition: str,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    known_constraint: known_constraint_fn = check_constraint,
    context: dict | None = None,
) -> float:
    """``-acquisition * known * unknown``."""
    x = np.asarray(x, dtype=float).reshape(-1)
    known = known_constraint(x)
    if known == 0:
        return 0.0
    unknown = predict_feasibility(classifier, x)
    y, s2 = predict_objective(model, x)
    t, d = x_obs.shape
    a = acquisition_value(acquisition, y, s2, float(np.max(y_obs)), t, d, context=context)
    return -a * known * unknown
