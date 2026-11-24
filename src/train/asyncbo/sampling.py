"""Inner acquisition search: port of ``getNextSamplingPoint{,ReduceMSE,ClfReduceMSE}.m``."""

from __future__ import annotations

import logging
from collections.abc import Callable

import cma
import numpy as np
from scipy.optimize import differential_evolution

from gp_omni.build.gpc import GPC
from gp_omni.build.gpr import GPR
from gp_omni.settings import load_defaults
from gp_omni.train.asyncbo.acquisition import (
    check_constraint,
    check_multiple_design_sites,
    known_constraint_fn,
    negative_acquisition,
    negative_classifier_mse,
    negative_mse,
)

DEFAULTS = load_defaults("bo")

log = logging.getLogger(__name__)


def minimize_cmaes(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    sigma: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    max_evals: int = DEFAULTS["cmaes"]["max_evals"],
    restarts: int = DEFAULTS["cmaes"]["restarts"],
) -> np.ndarray:
    """Bounded CMA-ES with restarts (``OPTS.Restarts = 3``); returns the best point seen.

    ``sigma`` is the per-coordinate initial step size, as in ``cmaes.m``.
    """
    best_x, best_f = np.clip(x0, lb, ub), np.inf

    def tracked(x):
        nonlocal best_x, best_f
        value = f(np.asarray(x))
        if value < best_f:
            best_x, best_f = np.asarray(x, dtype=float).copy(), value
        return value

    if len(lb) == 1:  # CMA-ES needs d >= 2
        differential_evolution(
            tracked, list(zip(lb, ub)), maxiter=max_evals // 15, seed=int(rng.integers(2**31))
        )
        return best_x
    cma.fmin(
        tracked,
        np.clip(x0, lb, ub),
        1.0,
        {
            "bounds": [list(lb), list(ub)],
            "CMA_stds": list(sigma),
            "maxfevals": max_evals,
            "maxiter": max_evals,
            "seed": int(rng.integers(1, 2**31)),
            "verbose": -9,
            "verb_log": 0,
        },
        restarts=restarts,
    )
    return best_x


def get_next_sampling_point(
    model: GPR,
    classifier: GPC | None,
    acquisition: str,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    rng: np.random.Generator,
    known_constraint: known_constraint_fn = check_constraint,
    max_evals: int = DEFAULTS["acquisition_search"]["max_evals"],
    max_retries: int = DEFAULTS["acquisition_search"]["max_retries"],
    relax_alpha: float | None = None,
) -> np.ndarray:
    """``getNextSamplingPoint.m``: maximise the penalised acquisition with CMA-ES.

    Starts from a random blend of the incumbent and a random point, and retries while the
    result violates the known constraint or duplicates a design site (capped at
    ``max_retries``, which the MATLAB ``while`` loop does not have).
    """
    x_lb, x_ub = np.asarray(x_lb, float), np.asarray(x_ub, float)
    d = len(x_ub)
    if acquisition == "MC":  # Monte Carlo random sampling
        log.info("Monte Carlo random sampling activated")
        return x_lb + rng.random(d) * (x_ub - x_lb)

    x_best = x_obs[int(np.argmax(y_obs))]
    sigma = (x_ub - x_lb) / 3

    def objective(x):
        return negative_acquisition(
            x, model, classifier, acquisition, x_obs, y_obs, known_constraint, relax_alpha
        )

    for attempt in range(max_retries):
        x_init = x_lb + rng.random(d) * (x_ub - x_lb)
        w = rng.random()
        x_next = minimize_cmaes(
            objective, w * x_best + (1 - w) * x_init, sigma, x_lb, x_ub, rng, max_evals
        )
        feasible = known_constraint(x_next) != 0
        duplicate = check_multiple_design_sites(x_next, x_obs)
        log.info("constraint_violated=%d duplicate_design_site=%d", not feasible, duplicate)
        if feasible and not duplicate:
            return x_next
    log.warning("no admissible point after %d tries; returning the last candidate", max_retries)
    return x_next


def get_next_sampling_point_reduce_mse(
    model: GPR,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    rng: np.random.Generator,
    max_evals: int = DEFAULTS["objective_exploration"]["max_evals"],
) -> np.ndarray:
    """``getNextSamplingPointReduceMSE.m``: explore where the objective model is most uncertain."""
    x_lb, x_ub = np.asarray(x_lb, float), np.asarray(x_ub, float)
    x0 = x_lb + rng.random(len(x_lb)) * (x_ub - x_lb)
    return minimize_cmaes(
        lambda x: negative_mse(x, model), x0, np.ones(len(x_lb)), x_lb, x_ub, rng, max_evals
    )


def get_next_sampling_point_classifier_reduce_mse(
    classifier: GPC | None,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    rng: np.random.Generator,
    max_evals: int = DEFAULTS["constraint_exploration"]["max_evals"],
) -> np.ndarray:
    """``getNextSamplingPointClfReduceMSE.m``: explore where the constraint is most uncertain.

    With no classifier yet (all simulations feasible) there is nothing to explore, so a
    uniform random point is returned.
    """
    x_lb, x_ub = np.asarray(x_lb, float), np.asarray(x_ub, float)
    x0 = x_lb + rng.random(len(x_lb)) * (x_ub - x_lb)
    if classifier is None:
        return x0
    return minimize_cmaes(
        lambda x: negative_classifier_mse(x, classifier),
        x0,
        np.ones(len(x_lb)),
        x_lb,
        x_ub,
        rng,
        max_evals,
    )
