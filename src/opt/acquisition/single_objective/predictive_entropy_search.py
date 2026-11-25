"""Predictive entropy search (PES) for a squared-exponential ARD Gaussian process.

Hernandez-Lobato, Hoffman & Ghahramani (2014), *Predictive entropy search for efficient
global optimization of black-box functions*, NeurIPS 27, 918-926 (arXiv 1406.2541)::

    PES(x) = H[p(y | D, x)] - E_{x*}[ H[p(y | D, x, x*)] ].

Implemented following section 2 of the paper, in the model's standardised target units:

1. **Sample optima** ``x*_k``: Thompson sampling from a random-Fourier-feature posterior
   sample of the GP, maximised with L-BFGS from the best of many random starts.
2. **Condition on ``x*``**: the GP is conditioned exactly on the data, ``grad f(x*) = 0`` and
   the off-diagonal Hessian ``= 0`` (derivatives of the SE kernel in closed form).
3. **Truncations** (expectation propagation, one factor per coordinate): ``f(x*)`` is at
   least the best posterior mean at the observed points, and every diagonal Hessian entry at
   ``x*`` is ``<= 0``. They depend on ``x*`` only, so they are computed once per proposal.
4. **Candidate**: the constraint ``f(x) <= f(x*)`` is moment-matched in closed form, giving
   the variance of ``y`` given ``x*``; the Gaussian entropies are then differenced.

Only ``cov='se_ard'`` is implemented. Other covariances raise ``NotImplementedError``
(``matern_*`` has too few derivatives at ``x*``; additive / multi-output / sparse models
are not implemented yet). ``x*`` is assumed to be an interior optimum, as in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import erfcx
from scipy.stats import norm

from gp_omni.build.gpr import GPR

JITTER = 1e-8  # relative to the signal variance
NOISE_FLOOR = 1e-6  # standardised units (targets have unit variance)


def check_supported(model: GPR) -> None:
    if (
        model.cov_kind != "se_ard"
        or model.additive
        or model.multitask
        or model.nu is not None
        or model.lik_name != "gaussian"
    ):
        raise NotImplementedError(
            "PES is not implemented yet for this model: it needs an exact, single-output "
            f"GPR with cov='se_ard' (got cov={model.cov_kind!r}, additive={model.additive!r}, "
            f"nu={model.nu!r}, multi-output/fidelity={model.multitask})"
        )


@dataclass
class se_model:
    """SE-ARD GP in standardised units, read from a fitted ``GPR``."""

    x: np.ndarray  # (n, d) training inputs
    r: np.ndarray  # (n,) standardised targets minus the constant mean
    mean: float
    lam: np.ndarray  # (d,) inverse squared length-scales
    sf2: float
    sn2: float
    sd: float  # target scale (original = standardised * sd)

    @classmethod
    def from_gpr(cls, model: GPR) -> se_model:
        check_supported(model)
        hyp = {k: v.detach().numpy() for k, v in model.hyp.items()}
        d = model._xa.shape[1]
        ell = np.exp(hyp["cov"][:d])
        mean = float(hyp["mean"][0])
        return cls(
            x=model._xa.numpy(),
            r=model._ys.numpy() - mean,
            mean=mean,
            lam=1.0 / ell**2,
            sf2=float(np.exp(2 * hyp["cov"][d])),
            sn2=float(np.exp(2 * hyp["lik"][0])),
            sd=float(model._sd[0]),
        )

    def k(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a[:, None, :] - b[None, :, :]
        return self.sf2 * np.exp(-0.5 * np.einsum("ijk,k->ij", diff**2, self.lam))


# ---------------------------------------------------------------- 1. optimum samples
def sample_optima(
    m: se_model,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 10,
    n_features: int = 500,
    n_starts: int = 1000,
) -> np.ndarray:
    """Maximisers of ``n_samples`` posterior function samples (random Fourier features)."""
    d = len(x_lb)
    w = rng.standard_normal((n_features, d)) * np.sqrt(m.lam)
    b = rng.uniform(0, 2 * np.pi, n_features)
    scale = math.sqrt(2 * m.sf2 / n_features)

    def phi(x):
        return scale * np.cos(x @ w.T + b)

    p = phi(m.x)
    sn2 = max(m.sn2, JITTER * m.sf2)  # a noise-free fit would make the feature system singular
    a = p.T @ p + sn2 * np.eye(n_features)
    chol = np.linalg.cholesky(a)
    theta_mean = cho_solve((chol, True), p.T @ m.r)
    bounds = list(zip(x_lb, x_ub, strict=True))
    optima = []
    for _ in range(n_samples):
        eps = rng.standard_normal(n_features)
        theta = theta_mean + math.sqrt(sn2) * np.linalg.solve(chol.T, eps)

        def neg(x, theta=theta):
            u = w @ x + b
            return -float(theta @ (scale * np.cos(u))), scale * (w.T @ (theta * np.sin(u)))

        starts = x_lb + rng.random((n_starts, d)) * (x_ub - x_lb)
        starts = np.vstack([starts, m.x])
        values = phi(starts) @ theta
        best = None
        for x0 in starts[np.argsort(-values)[:5]]:
            res = minimize(neg, x0, jac=True, method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        optima.append(best.x)
    return np.array(optima)


# ------------------------------------------------- 2. covariances of derivative functionals
def _pairs(d: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def cross_with_observations(m: se_model, x: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """``cov(f(x_a), [f(xs), grad_i f(xs), H_ij f(xs) (i<j)])``; returns ``(len(x), 1+d+P)``."""
    d, lam = len(xs), m.lam
    kx = m.k(x, xs[None, :])[:, 0]
    u = x - xs
    cols = [kx[:, None], kx[:, None] * lam * u]
    cols += [(kx * lam[i] * lam[j] * u[:, i] * u[:, j])[:, None] for i, j in _pairs(d)]
    return np.hstack(cols)


def cross_with_curvature(m: se_model, x: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """``cov(f(x_a), [f(xs), H_ii f(xs)])``; returns ``(len(x), 1+d)``."""
    lam = m.lam
    kx = m.k(x, xs[None, :])[:, 0]
    u = x - xs
    return np.hstack([kx[:, None], kx[:, None] * (lam**2 * u**2 - lam)])


def prior_blocks(m: se_model, xs: np.ndarray):
    """Prior covariances at the optimum: ``K_gg/H`` (observed), ``K_Oq`` and ``K_qq``.

    Observed functionals ``O = [y(X), grad f(xs), H_ij f(xs) (i<j)]``; unknowns
    ``q = [f(xs), H_ii f(xs)]``. Returns ``K_OO`` (with noise / jitter) and ``K_Oq``, ``K_qq``.
    """
    d, lam, sf2 = len(xs), m.lam, m.sf2
    n, p = len(m.x), len(_pairs(d))
    k_xo = cross_with_observations(m, m.x, xs)  # (n, 1+d+p): f(xs) column first
    k_xq = cross_with_curvature(m, m.x, xs)
    no = n + d + p
    k_oo = np.zeros((no, no))
    k_oo[:n, :n] = m.k(m.x, m.x) + m.sn2 * np.eye(n)
    k_oo[:n, n:] = k_xo[:, 1:]
    k_oo[n:, :n] = k_xo[:, 1:].T
    k_oo[n : n + d, n : n + d] = sf2 * np.diag(lam)
    k_oo[n + d :, n + d :] = sf2 * np.diag([lam[i] * lam[j] for i, j in _pairs(d)])
    k_oo += JITTER * sf2 * np.eye(no)
    k_oq = np.zeros((no, 1 + d))
    k_oq[:n] = k_xq
    k_qq = np.zeros((1 + d, 1 + d))
    k_qq[0, 0] = sf2
    k_qq[0, 1:] = k_qq[1:, 0] = -lam * sf2
    k_qq[1:, 1:] = sf2 * (2 * np.diag(lam**2) + np.outer(lam, lam))
    return k_oo, k_oq, k_qq


# ------------------------------------------------------------------ 3. EP truncations
def truncated_moments(mean: float, var: float, bound: float, sign: float) -> tuple[float, float]:
    """Mean / variance of ``N(mean, var)`` truncated to ``sign * (z - bound) >= 0``."""
    s = math.sqrt(var)
    alpha = sign * (mean - bound) / s
    if alpha < -100:
        # Tail expansions of the excess mean and variance avoid subtracting nearly
        # equal numbers in both mean + s*lambda and 1 - lambda*(alpha + lambda).
        inv = -1 / alpha
        u = inv**2
        excess = inv * (1 + u * (-2 + u * (10 + u * (-74 + u * 706))))
        factor = u * (1 + u * (-6 + u * (50 + u * (-518 + u * 6354))))
        return bound + sign * s * excess, var * factor
    lam = (
        math.sqrt(2 / math.pi) / erfcx(-alpha / math.sqrt(2))
        if alpha <= 0
        else norm.pdf(alpha) / norm.cdf(alpha)
    )
    return mean + sign * s * lam, var * max(1 - lam * (alpha + lam), 1e-12)


def ep_truncations(
    mean: np.ndarray,
    cov: np.ndarray,
    bounds: np.ndarray,
    signs: np.ndarray,
    sweeps: int = 20,
    damping: float = 0.5,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """EP site parameters ``(nu, tau)`` for ``sign_j * (q_j - bound_j) >= 0`` on ``N(mean, cov)``."""
    dim = len(mean)
    prec = np.linalg.inv(cov + 1e-12 * np.eye(dim))
    nu, tau = np.zeros(dim), np.zeros(dim)
    for _ in range(sweeps):
        shift = 0.0
        for j in range(dim):
            sigma = np.linalg.inv(prec + np.diag(tau))
            mu = sigma @ (prec @ mean + nu)
            tau_c = 1 / sigma[j, j] - tau[j]
            if tau_c <= 1e-12:
                continue
            nu_c = mu[j] / sigma[j, j] - nu[j]
            m_hat, v_hat = truncated_moments(nu_c / tau_c, 1 / tau_c, bounds[j], signs[j])
            tau_new = max(1 / v_hat - tau_c, 0.0)
            nu_new = m_hat / v_hat - nu_c
            tau_next = (1 - damping) * tau[j] + damping * tau_new
            nu_next = (1 - damping) * nu[j] + damping * nu_new
            shift = max(shift, abs(tau_next - tau[j]) + abs(nu_next - nu[j]))
            tau[j], nu[j] = tau_next, nu_next
        if shift < tol:
            break
    return nu, tau


@dataclass
class optimum_state:
    """Everything about one sampled ``x*`` that does not depend on the candidate."""

    xs: np.ndarray
    k_inv: np.ndarray  # K_OO^{-1}
    beta: np.ndarray  # K_OO^{-1} r_O
    a_q: np.ndarray  # K_OO^{-1} K_Oq
    k_qq: np.ndarray
    s_q: np.ndarray  # cov of q after conditioning on O
    m_q: np.ndarray  # mean of q after conditioning on O (standardised, mean included)
    nu: np.ndarray
    tau: np.ndarray


def build_state(m: se_model, xs: np.ndarray, f_best: float) -> optimum_state:
    d = len(xs)
    n = len(m.x)
    k_oo, k_oq, k_qq = prior_blocks(m, xs)
    r_o = np.concatenate([m.r, np.zeros(k_oo.shape[0] - n)])
    chol = cho_factor(k_oo, lower=True)
    k_inv = cho_solve(chol, np.eye(k_oo.shape[0]))
    beta = k_inv @ r_o
    a_q = k_inv @ k_oq
    s_q = k_qq - k_oq.T @ a_q
    s_q = (s_q + s_q.T) / 2
    m_q = k_oq.T @ beta + np.concatenate([[m.mean], np.zeros(d)])
    bounds = np.concatenate([[f_best], np.zeros(d)])
    signs = np.concatenate([[1.0], -np.ones(d)])  # f(x*) >= best, H_ii <= 0
    nu, tau = ep_truncations(m_q, s_q, bounds, signs)
    return optimum_state(xs, k_inv, beta, a_q, k_qq, s_q, m_q, nu, tau)


def prepare(
    model: GPR,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 10,
) -> dict:
    """Per-proposal PES state: the model snapshot and one :class:`optimum_state` per ``x*``."""
    m = se_model.from_gpr(model)
    fmu = model.predict(torch.as_tensor(m.x))["fmu"].numpy()
    f_best = float((fmu.max() - model._mu[0].item()) / m.sd)
    optima = sample_optima(m, np.asarray(x_lb, float), np.asarray(x_ub, float), rng, n_samples)
    return {"pes_model": m, "pes_states": [build_state(m, xs, f_best) for xs in optima]}


# ------------------------------------------------------------- 4. the acquisition value
def conditional_variance(m: se_model, state: optimum_state, x: np.ndarray) -> float:
    """Variance of ``f(x)`` given the data, ``x*`` and its constraints, then ``f(x) <= f(x*)``."""
    xs = state.xs
    x2 = x[None, :]
    k_xo = cross_with_observations(m, x2, xs)
    k_xo = np.hstack([m.k(x2, m.x), k_xo[:, 1:]])[0]
    k_xq = cross_with_curvature(m, x2, xs)[0]
    mean_x = m.mean + k_xo @ state.beta
    var_x = m.sf2 - k_xo @ state.k_inv @ k_xo
    c_xq = k_xq - k_xo @ state.a_q
    # joint Gaussian of (f(x), q) after conditioning on O, then the EP site factors on q
    dim = len(state.m_q)
    joint = np.empty((dim + 1, dim + 1))
    joint[0, 0] = var_x
    joint[0, 1:] = joint[1:, 0] = c_xq
    joint[1:, 1:] = state.s_q
    mean = np.concatenate([[mean_x], state.m_q])
    t = np.diag(state.tau)
    sq = state.s_q
    w = np.linalg.solve(np.eye(dim) + t @ sq, t)  # (T^-1 + S)^-1 without inverting T
    cp = joint[:, 1:]  # C P^T
    post_cov = joint - cp @ w @ cp.T
    sigma_p = cp - cp @ w @ sq  # Sigma' P^T part: C P^T - C P^T (T^-1+S)^-1 S
    post_mean = mean - cp @ (w @ state.m_q) + sigma_p @ state.nu
    # truncation f(x) <= f(x*)
    v_x, v_s, c = post_cov[0, 0], post_cov[1, 1], post_cov[0, 1]
    mu_t = post_mean[1] - post_mean[0]
    s = v_s + v_x - 2 * c
    if s <= 1e-12:
        return max(v_x, 1e-12)
    _, var_t = truncated_moments(mu_t, s, 0.0, 1.0)
    cov_ft = c - v_x
    return max(v_x - (cov_ft / s) ** 2 * (s - var_t), 1e-12)


def predictive_entropy_search(x: np.ndarray, model: GPR, context: dict) -> float:
    """PES at ``x``; ``context`` comes from :func:`prepare`.

    Both entropies use the noise variance floored at ``NOISE_FLOOR`` (standardised units), so
    a noise-free fit does not make the difference of two ``log 0`` terms meaningless. The
    conditional variance is capped at the prior one: conditioning cannot add uncertainty, so
    anything above it is approximation error, and PES stays non-negative.
    """
    m: se_model = context["pes_model"]
    x = np.asarray(x, float).reshape(-1)
    out = model.predict(torch.as_tensor(x[None, :]))
    prior_f = float(out["fs2"][0]) / m.sd**2
    noise = max(m.sn2, NOISE_FLOOR)
    cond = [min(conditional_variance(m, st, x), prior_f) + noise for st in context["pes_states"]]
    value = 0.5 * math.log(prior_f + noise) - 0.5 * float(np.mean(np.log(cond)))
    return max(value, 0.0)
