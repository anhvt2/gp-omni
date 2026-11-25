"""Knowledge gradient (KG) for a Gaussian-process belief over a discrete candidate set.

``KG(x) = E[max_a mu_{n+1}(a) | x_{n+1} = x] - max_a mu_n(a)`` (one-step look-ahead), with
``a`` ranging over a candidate set ``A``.

* Frazier, Powell & Dayanik (2009), *The knowledge-gradient policy for correlated normal
  beliefs*, INFORMS J. Computing 21(4):599-613 (the discrete algorithm below).
* Scott, Frazier & Powell (2011), *The correlated knowledge gradient for simulation
  optimization of continuous parameters using Gaussian process regression*, SIAM J. Optim.
  21(3) (the GP version).

After observing ``y = f(x) + noise``, the posterior mean at ``a`` moves by
``sigma_tilde(a, x) Z`` with ``Z ~ N(0, 1)`` and
``sigma_tilde(a, x) = cov(f(a), f(x)) / sqrt(var f(x) + noise)``. KG is then the expected
maximum of the lines ``mu(a) + sigma_tilde(a) Z``, which is exact (see
:func:`expected_max_of_lines`). Parallel runs need nothing more: pending points are
hallucinated into the model before this function is evaluated.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm, qmc

from gp_omni.build.gpr import GPR


def check_supported(model: GPR) -> None:
    if getattr(model, "multitask", False) or getattr(model, "nu", None) is not None:
        raise NotImplementedError(
            "KG is not implemented yet for multi-output, multi-fidelity or sparse GPs"
        )


def expected_max_of_lines(a: np.ndarray, b: np.ndarray) -> float:
    """``E[max_i (a_i + b_i Z)]`` for ``Z ~ N(0, 1)``.

    Builds the upper envelope of the lines (sort by slope, drop dominated lines) and sums
    the Gaussian integral over each piece (Frazier et al. 2009, algorithm 1).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    order = np.lexsort((a, b))  # by slope, ties by intercept
    a, b = a[order], b[order]
    keep = np.append(b[1:] != b[:-1], True)  # equal slopes: the highest intercept wins
    a, b = a[keep], b[keep]
    hull_a: list[float] = []
    hull_b: list[float] = []
    cross: list[float] = []  # cross[k] is where line k+1 overtakes line k
    for ai, bi in zip(a, b, strict=True):
        while hull_a:
            c = (hull_a[-1] - ai) / (bi - hull_b[-1])
            if cross and c <= cross[-1]:
                hull_a.pop(), hull_b.pop(), cross.pop()
            else:
                break
        if hull_a:
            cross.append((hull_a[-1] - ai) / (bi - hull_b[-1]))
        hull_a.append(ai)
        hull_b.append(bi)
    edges = np.concatenate([[-np.inf], cross, [np.inf]])
    ha, hb = np.array(hull_a), np.array(hull_b)
    cdf, pdf = norm.cdf(edges), norm.pdf(edges)
    return float(np.sum(ha * (cdf[1:] - cdf[:-1]) + hb * (pdf[:-1] - pdf[1:])))


def candidate_set(
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    rng: np.random.Generator,
    n_candidates: int = 128,
    n_incumbents: int = 20,
) -> np.ndarray:
    """Scrambled Sobol points in the box plus the best observed points."""
    x_lb, x_ub = np.asarray(x_lb, float), np.asarray(x_ub, float)
    sobol = qmc.Sobol(d=len(x_lb), scramble=True, seed=int(rng.integers(2**31)))
    unit = sobol.random(n_candidates)
    best = np.asarray(x_obs)[np.argsort(-np.asarray(y_obs))[:n_incumbents]]
    return np.vstack([x_lb + unit * (x_ub - x_lb), best])


def knowledge_gradient(x: np.ndarray, model: GPR, candidates: np.ndarray) -> float:
    """KG of sampling ``x``; ``candidates`` is the belief-evaluation set ``A``."""
    xs = torch.as_tensor(np.vstack([candidates, np.asarray(x, float).reshape(1, -1)]))
    out = model.predict(xs, full_cov=True)
    mu = out["fmu"].numpy()
    cov = out["fcov"].numpy()
    noise = float(out["ys2"][-1] - out["fs2"][-1])
    total = cov[-1, -1] + noise
    if total <= 0:
        return 0.0
    sigma_tilde = cov[:, -1] / np.sqrt(total)
    return max(expected_max_of_lines(mu, sigma_tilde) - float(mu.max()), 0.0)
