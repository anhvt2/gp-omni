"""Maximum-likelihood training and posterior factor assembly for co-kriging."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor

from gp_omni.build.cokriging import _Level
from gp_omni.build.correlation import correlation
from gp_omni.build.linalg import _solve_lower, _solve_upper_t
from gp_omni.build.regression import regression_matrix

_EPS = float(torch.finfo(torch.float64).eps)
_BIG = 1e10


def _level_objective(
    name: str, regression: str, X: Tensor, y: Tensor, theta: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Concentrated negative log-likelihood of one level.

    Returns ``(nll, sigma2, beta, L, gamma)``. ``sigma2 = RSS / n`` and the
    trend coefficients are profiled out (GLS), as in ``marginalLikelihood.m``.
    """
    n = X.shape[0]
    psi = correlation(name, theta, X, X)
    psi = psi + (n + 10) * _EPS * torch.eye(n, dtype=X.dtype)
    L = torch.linalg.cholesky(psi)
    F = regression_matrix(regression, X)
    Yt = _solve_lower(L, y.unsqueeze(-1))
    if F.shape[-1] > 0:
        Ft = _solve_lower(L, F)
        Q, R = torch.linalg.qr(Ft)
        beta = torch.linalg.solve_triangular(R, Q.T @ Yt, upper=True)
        resid = Yt - Ft @ beta
    else:
        beta = Yt.new_zeros(0, 1)
        resid = Yt
    sigma2 = (resid**2).sum() / n
    nll = 0.5 * (n * torch.log(sigma2) + 2.0 * torch.log(torch.diagonal(L)).sum())
    gamma = _solve_upper_t(L, resid)
    return nll, sigma2, beta.squeeze(-1), L, gamma.squeeze(-1)


def _minimize(fun, x0: np.ndarray, lower: np.ndarray, upper: np.ndarray, restarts: int, seed: int):
    """Multi-start bounded L-BFGS-B on a torch objective ``fun(Tensor) -> Tensor``."""

    def closure(x: np.ndarray):
        t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        try:
            val = fun(t)
            val.backward()
            assert t.grad is not None
            g = t.grad.numpy().copy()
            v = float(val.detach())
            if not np.isfinite(v) or not np.all(np.isfinite(g)):
                return _BIG, np.zeros_like(x)
            return v, g
        except (torch.linalg.LinAlgError, RuntimeError):
            return _BIG, np.zeros_like(x)

    rng = np.random.default_rng(seed)
    starts = [np.clip(x0, lower, upper)]
    starts += [rng.uniform(lower, upper) for _ in range(restarts)]
    best = None
    for s in starts:
        res = minimize(closure, s, jac=True, method="L-BFGS-B", bounds=list(zip(lower, upper)))
        if best is None or res.fun < best.fun:
            best = res
    if best is None or best.fun >= _BIG:
        raise RuntimeError(
            "hyperparameter optimisation failed (ill-conditioned correlation matrix)"
        )
    return best.x


def fit_ar1(model, hp0, hp_bounds, rho0, rho_bounds, num_restarts, seed, hyperparameters):
    model._fitted = False
    correlation, regression = model.correlation_name, model.regression
    dt = torch.float64
    n_hp = model.dim
    fixed = hyperparameters is not None
    hp_fixed: dict = hyperparameters or {}
    lo, hi = hp_bounds
    x_lo = model.X[0]

    # ---- level 1 ----
    h0 = np.broadcast_to(np.asarray(hp0, dtype=float), (n_hp,)).copy()
    if fixed:
        th1 = torch.as_tensor(hp_fixed["theta1"], dtype=dt).reshape(-1)
    else:
        th1 = torch.tensor(
            _minimize(
                lambda t: _level_objective(correlation, regression, x_lo, model.Y[0], t)[0],
                h0,
                np.full(n_hp, lo),
                np.full(n_hp, hi),
                num_restarts,
                seed,
            ),
            dtype=dt,
        )
    with torch.no_grad():
        _, s1, b1, L1, g1 = _level_objective(correlation, regression, x_lo, model.Y[0], th1)
    model.level1 = _Level(x_lo, th1, 0.0, s1, b1, L1, g1)

    # ---- level 2 (rho, theta2) ----
    x_hi = model.X[1]
    yc = model._level_mean(model.level1, x_hi)

    def obj2(v: Tensor) -> Tensor:
        r, th = v[0], v[1:]
        return _level_objective(correlation, regression, x_hi, model.Y[1] - r * yc, th)[0]

    if fixed:
        rho = float(hp_fixed["rho"])
        th2 = torch.as_tensor(hp_fixed["theta2"], dtype=dt).reshape(-1)
    else:
        x0 = np.concatenate([[rho0], h0])
        lb = np.concatenate([[rho_bounds[0]], np.full(n_hp, lo)])
        ub = np.concatenate([[rho_bounds[1]], np.full(n_hp, hi)])
        sol = _minimize(obj2, x0, lb, ub, num_restarts, seed + 1)
        rho, th2 = float(sol[0]), torch.tensor(sol[1:], dtype=dt)
    with torch.no_grad():
        _, s2, b2, L2, g2 = _level_objective(
            correlation, regression, x_hi, model.Y[1] - rho * yc, th2
        )
    model.level2 = _Level(x_hi, th2, rho, s2, b2, L2, g2)
    model.rho = rho
    assemble_ar1(model)

    model._fitted = True
    return model


def assemble_ar1(model):
    l1, l2, rho = model.level1, model.level2, model.rho
    s1, s2 = l1.sigma2, l2.sigma2
    X1, X2 = model.X
    n1, n2 = X1.shape[0], X2.shape[0]
    c = model.correlation_name
    with torch.no_grad():
        S11 = s1 * correlation(c, l1.theta, X1, X1)
        S12 = rho * s1 * correlation(c, l1.theta, X1, X2)
        S22 = rho**2 * s1 * correlation(c, l1.theta, X2, X2) + s2 * correlation(c, l2.theta, X2, X2)
        S = torch.cat([torch.cat([S11, S12], 1), torch.cat([S12.T, S22], 1)], 0)
        nug = torch.cat(
            [
                s1 * (n1 + 10) * _EPS * torch.ones(n1, dtype=S.dtype),
                s2 * (n2 + 10) * _EPS * torch.ones(n2, dtype=S.dtype),
            ]
        )
        S = S + torch.diag(nug)
        F1 = regression_matrix(model.regression, X1)
        F2 = regression_matrix(model.regression, X2)
        p = F1.shape[-1]
        F = torch.cat(
            [
                torch.cat([F1, F1.new_zeros(n1, p)], 1),
                torch.cat([rho * F2, F2], 1),
            ],
            0,
        )
        y = torch.cat(model.Y).unsqueeze(-1)
        L = torch.linalg.cholesky(S)
        Yt = _solve_lower(L, y)
        if p > 0:
            Ft = _solve_lower(L, F)
            Q, R = torch.linalg.qr(Ft)
            beta = torch.linalg.solve_triangular(R, Q.T @ Yt, upper=True)
            resid = Yt - Ft @ beta
        else:
            Ft, R, beta, resid = F, None, y.new_zeros(0, 1), Yt
        model._L, model._Ft, model._R = L, Ft, R
        model.beta = beta.squeeze(-1)
        model._gamma = _solve_upper_t(L, resid).squeeze(-1)
        model.sigma2 = rho**2 * s1 + s2  # prior variance of the high-fidelity process


def fit_dag(
    model,
    hp0,
    hp_bounds,
    rho0,
    rho_bounds,
    log_variance_bounds,
    num_restarts,
    seed,
    hyperparameters,
):
    model._fitted = False
    correlation, regression = model.correlation_name, model.regression
    order = model.topological_order
    size = model.num_nodes * model.dim
    if hyperparameters is not None:
        theta = torch.as_tensor(hyperparameters["theta"], dtype=torch.float64, device="cpu")
        if theta.shape == (model.num_nodes, 1):
            theta = theta.expand(-1, model.dim).clone()
        rho = torch.as_tensor(hyperparameters["rho"], dtype=torch.float64, device="cpu")
        variances = torch.as_tensor(hyperparameters["sigma2"], dtype=torch.float64, device="cpu")
    else:
        bounds = (hp_bounds, rho_bounds, log_variance_bounds)
        if any(len(b) != 2 or not np.isfinite(b).all() or b[0] >= b[1] for b in bounds):
            raise ValueError("hyperparameter bounds must be finite increasing pairs")
        if not isinstance(num_restarts, Integral) or num_restarts < 0:
            raise ValueError("num_restarts must be a nonnegative integer")
        h0 = np.broadcast_to(np.asarray(hp0, dtype=float), (model.dim,)).copy()
        if not np.isfinite(h0).all() or not np.isfinite(rho0):
            raise ValueError("initial hyperparameters must be finite")
        h0 = np.clip(h0, *hp_bounds)
        r0 = float(np.clip(rho0, *rho_bounds))
        # Sequential parent means supply initial scales only. Fitting below is joint.
        initial_levels = {}
        A0 = model._loadings(torch.full((len(model.edges),), r0, dtype=torch.float64))
        scales = torch.zeros(model.num_nodes, dtype=torch.float64)
        for j in order:
            residual = model.Y[j].clone()
            for q in order[: order.index(j)]:
                if A0[j, q] != 0:
                    lvl = initial_levels[q]
                    mu = model._discrepancy_mean(lvl, model.X[j])
                    residual = residual - A0[j, q] * mu
            th = torch.tensor(h0)
            with torch.no_grad():
                _, s, b, L, g = _level_objective(correlation, regression, model.X[j], residual, th)
            scales[j] = s.clamp_min(max(float(model.Y[j].square().mean()), 1.0) * 1e-10)
            initial_levels[j] = _Level(model.X[j], th, r0, scales[j], b, L, g)

        def objective(v: Tensor) -> Tensor:
            th = v[:size].reshape(model.num_nodes, model.dim)
            r = v[size : size + len(model.edges)]
            s = scales * torch.pow(10.0, v[size + len(model.edges) :])
            L, _, _, _, resid = factor_dag(model, th, r, s)
            return 0.5 * resid.square().sum() + torch.log(L.diagonal()).sum()

        x0 = np.concatenate(
            [
                np.tile(h0, model.num_nodes),
                np.full(len(model.edges), r0),
                np.zeros(model.num_nodes),
            ]
        )
        lower = np.concatenate(
            [
                np.full(size, hp_bounds[0]),
                np.full(len(model.edges), rho_bounds[0]),
                np.full(model.num_nodes, log_variance_bounds[0]),
            ]
        )
        upper = np.concatenate(
            [
                np.full(size, hp_bounds[1]),
                np.full(len(model.edges), rho_bounds[1]),
                np.full(model.num_nodes, log_variance_bounds[1]),
            ]
        )
        sol = torch.tensor(_minimize(objective, x0, lower, upper, num_restarts, seed))
        theta = sol[:size].reshape(model.num_nodes, model.dim)
        rho = sol[size : size + len(model.edges)]
        variances = scales * torch.pow(10.0, sol[size + len(model.edges) :])
    if (
        theta.shape != (model.num_nodes, model.dim)
        or rho.shape != (len(model.edges),)
        or variances.shape != (model.num_nodes,)
    ):
        raise ValueError("invalid theta, rho, or sigma2 shape")
    if (
        not all(torch.isfinite(t).all() for t in (theta, rho, variances))
        or not (variances > 0).all()
    ):
        raise ValueError("hyperparameters must be finite and sigma2 strictly positive")
    model.register_buffer("theta", theta.detach().clone())
    model.register_buffer("rho", rho.detach().clone())
    model.register_buffer("variances", variances.detach().clone())
    with torch.no_grad():
        L, Ft, R, beta, resid = factor_dag(model, model.theta, model.rho, model.variances)
        model.register_buffer("loadings", model._loadings(model.rho))
        model.register_buffer("_L", L)
        model.register_buffer("_Ft", Ft)
        model.register_buffer("_R", R)
        model.register_buffer("beta", beta.squeeze(-1))
        model.register_buffer("_gamma", _solve_upper_t(L, resid).squeeze(-1))

    model._fitted = True
    return model


def factor_dag(model, theta, rho, variances):
    A = model._loadings(rho)
    S = torch.cat(
        [
            torch.cat(
                [model._covariance(i, j, x, z, A, theta, variances) for j, z in enumerate(model.X)],
                dim=-1,
            )
            for i, x in enumerate(model.X)
        ],
        dim=-2,
    )
    # Relative numerical jitter, not observation noise.
    S = S + torch.diag(S.diagonal() * (S.shape[0] + 10) * _EPS)
    F = torch.cat([model._design(i, x, A) for i, x in enumerate(model.X)])
    L = torch.linalg.cholesky(S)
    Yt = _solve_lower(L, torch.cat(model.Y).unsqueeze(-1))
    Ft = _solve_lower(L, F)
    if F.shape[-1]:
        Q, R = torch.linalg.qr(Ft)
        beta = torch.linalg.solve_triangular(R, Q.T @ Yt, upper=True)
    else:
        R, beta = F.new_zeros(0, 0), F.new_zeros(0, 1)
    return L, Ft, R, beta, Yt - Ft @ beta
