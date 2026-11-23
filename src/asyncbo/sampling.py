"""Inner acquisition search: port of ``getNextSamplingPoint{,ReduceMSE,ClfReduceMSE}.m``."""

from __future__ import annotations

import logging
from collections.abc import Callable

import cma
import numpy as np
from scipy.optimize import differential_evolution

from gp_omni.asyncbo.acquisition import (
    KnownConstraint,
    check_constraint,
    check_multiple_design_sites,
    negative_acquisition,
    negative_clf_mse,
    negative_mse,
)
from gp_omni.sparse.omni import OmniGP, OmniGPC

log = logging.getLogger(__name__)


def minimize_cmaes(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    sigma: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    max_evals: int = 250,
    restarts: int = 3,
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
    model: OmniGP,
    clf: OmniGPC | None,
    acquisition: str,
    xLB: np.ndarray,
    xUB: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    known_constraint: KnownConstraint = check_constraint,
    max_evals: int = 250,
    max_retries: int = 100,
) -> np.ndarray:
    """``getNextSamplingPoint.m``: maximise the penalised acquisition with CMA-ES.

    Starts from a random blend of the incumbent and a random point, and retries while the
    result violates the known constraint or duplicates a design site (capped at
    ``max_retries``, which the MATLAB ``while`` loop does not have).
    """
    xLB, xUB = np.asarray(xLB, float), np.asarray(xUB, float)
    d = len(xUB)
    if acquisition == "MC":  # Monte Carlo random sampling
        log.info("Monte Carlo random sampling activated")
        return xLB + rng.random(d) * (xUB - xLB)

    x_best = S[int(np.argmax(Y))]
    sigma = (xUB - xLB) / 3

    def objective(x):
        return negative_acquisition(x, model, clf, acquisition, S, Y, known_constraint)

    for attempt in range(max_retries):
        x_init = xLB + rng.random(d) * (xUB - xLB)
        w = rng.random()
        x_next = minimize_cmaes(
            objective, w * x_best + (1 - w) * x_init, sigma, xLB, xUB, rng, max_evals
        )
        feasible = known_constraint(x_next) != 0
        duplicate = check_multiple_design_sites(x_next, S)
        log.info("constraintViolator=%d multipleDesignSitesViolator=%d", not feasible, duplicate)
        if feasible and not duplicate:
            return x_next
    log.warning("no admissible point after %d tries; returning the last candidate", max_retries)
    return x_next


def get_next_sampling_point_reduce_mse(
    model: OmniGP, xLB: np.ndarray, xUB: np.ndarray, rng: np.random.Generator, max_evals: int = 100
) -> np.ndarray:
    """``getNextSamplingPointReduceMSE.m``: explore where the objective model is most uncertain."""
    xLB, xUB = np.asarray(xLB, float), np.asarray(xUB, float)
    x0 = xLB + rng.random(len(xLB)) * (xUB - xLB)
    return minimize_cmaes(
        lambda x: negative_mse(x, model), x0, np.ones(len(xLB)), xLB, xUB, rng, max_evals
    )


def get_next_sampling_point_clf_reduce_mse(
    clf: OmniGPC | None,
    xLB: np.ndarray,
    xUB: np.ndarray,
    rng: np.random.Generator,
    max_evals: int = 100,
) -> np.ndarray:
    """``getNextSamplingPointClfReduceMSE.m``: explore where the constraint is most uncertain.

    With no classifier yet (all simulations feasible) there is nothing to explore, so a
    uniform random point is returned.
    """
    xLB, xUB = np.asarray(xLB, float), np.asarray(xUB, float)
    x0 = xLB + rng.random(len(xLB)) * (xUB - xLB)
    if clf is None:
        return x0
    return minimize_cmaes(
        lambda x: negative_clf_mse(x, clf), x0, np.ones(len(xLB)), xLB, xUB, rng, max_evals
    )
