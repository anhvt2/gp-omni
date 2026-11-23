"""Two-fidelity AR1 co-kriging (Kennedy & O'Hagan 2000; Forrester et al. 2007).

Port of the owner's MATLAB ``ooDACE/@CoKriging`` (Couckuyt et al., ooDACE toolbox; the
multi-fidelity extension is the repository owner's work). The model is

    f_lo(x) ~ GP(F(x) b1, s1 * Psi1(x, x'))
    f_hi(x) = rho * f_lo(x) + delta(x),   delta ~ GP(F(x) b2, s2 * Psi2(x, x'))

Fitting is sequential, as in ``@CoKriging/fit.m``:

1. fit level 1 on the low-fidelity data (concentrated likelihood, ``s1`` profiled out);
2. fit level 2 on ``y_hi - rho * mu_lo(X_hi)`` over ``(rho, theta2)``;
3. assemble the *joint* covariance of ``[y_lo; y_hi]`` and compute a generalised
   least-squares mean (``b = [b1; b2]``), used for prediction.

Differences from the MATLAB code, deliberately:

* exactly two fidelity levels (same limit as ooDACE);
* the predictive-variance regression term uses ``(Ft' Ft)^-1 = R^-1 R^-T``; ooDACE solves with
  ``R`` instead of ``R'``, which only differs when the regression basis has > 1 column;
* hyperparameters are fitted with autograd + L-BFGS-B multi-start instead of SQPLab.

Inputs are used as given (no normalisation), like ooDACE. Scale them to a comparable range
first (e.g. the unit cube), since the default bounds assume that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from scipy.optimize import minimize
from torch import Tensor

from gp_omni.correlation import CORRELATIONS, correlation
from gp_omni.regression import regression_matrix

_EPS = float(torch.finfo(torch.float64).eps)
_BIG = 1e10


def _solve_lower(L: Tensor, b: Tensor) -> Tensor:
    return torch.linalg.solve_triangular(L, b, upper=False)


def _solve_upper_t(L: Tensor, b: Tensor) -> Tensor:
    """Solve ``L^T x = b`` for lower-triangular ``L``."""
    return torch.linalg.solve_triangular(L.transpose(-1, -2), b, upper=True)


@dataclass
class _Level:
    """One fitted universal-kriging level (MATLAB ``BasicGaussianProcess`` without noise)."""

    X: Tensor
    theta: Tensor  # log10 precisions
    rho: float
    sigma2: Tensor
    beta: Tensor
    L: Tensor
    gamma: Tensor  # Psi^-1 (y - F beta)


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


class AR1CoKriging(Model):
    """Two-fidelity co-kriging with a BoTorch-compatible posterior for the high fidelity.

    Parameters
    ----------
    train_X, train_Y
        ``[low, high]`` lists of ``(n_i, d)`` inputs and ``(n_i,)`` / ``(n_i, 1)`` outputs.
        The designs need not be nested.
    correlation
        ``"gauss"``, ``"exp"``, ``"matern32"`` or ``"matern52"``.
    regression
        ``"none"`` (zero-mean, ooDACE ``oodacefit`` default), ``"regpoly0"`` (constant),
        ``"regpoly1"`` or ``"regpoly2"``. Shared by both levels.
    hp0, hp_bounds
        Initial ``log10`` precision (scalar or per-dim) and ``(lo, hi)`` bounds.
    rho0, rho_bounds
        Initial scaling factor and its bounds (ooDACE defaults: 5 and ``[0.1, 5]``).
    hyperparameters
        Optional ``{"theta1", "theta2", "rho"}``; if given, no optimisation is run.
    """

    def __init__(
        self,
        train_X: list[Tensor],
        train_Y: list[Tensor],
        correlation: str = "matern32",
        regression: str = "none",
        hp0: float | list[float] = 0.5,
        hp_bounds: tuple[float, float] = (-2.0, 2.0),
        rho0: float = 5.0,
        rho_bounds: tuple[float, float] = (0.1, 5.0),
        num_restarts: int = 4,
        seed: int = 0,
        hyperparameters: dict | None = None,
    ) -> None:
        super().__init__()
        if len(train_X) != 2 or len(train_Y) != 2:
            raise ValueError("AR1CoKriging supports exactly two fidelity levels [low, high]")
        if correlation not in CORRELATIONS:
            raise ValueError(f"unknown correlation {correlation!r}")
        self.correlation_name = correlation
        self.regression = regression
        dt = torch.float64
        self.X = [torch.as_tensor(x, dtype=dt) for x in train_X]
        self.Y = [torch.as_tensor(y, dtype=dt).reshape(-1) for y in train_Y]
        for x, y in zip(self.X, self.Y):
            if x.ndim != 2 or x.shape[0] != y.shape[0]:
                raise ValueError("each X must be (n, d) with matching Y length")
        if self.X[0].shape[1] != self.X[1].shape[1]:
            raise ValueError("both fidelity levels must share the input dimension")
        self.dim = self.X[0].shape[1]

        n_hp = self.dim
        fixed = hyperparameters is not None
        hp_fixed: dict = hyperparameters or {}
        lo, hi = hp_bounds
        x_lo = self.X[0]

        # ---- level 1 ----
        h0 = np.broadcast_to(np.asarray(hp0, dtype=float), (n_hp,)).copy()
        if fixed:
            th1 = torch.as_tensor(hp_fixed["theta1"], dtype=dt).reshape(-1)
        else:
            th1 = torch.tensor(
                _minimize(
                    lambda t: _level_objective(correlation, regression, x_lo, self.Y[0], t)[0],
                    h0,
                    np.full(n_hp, lo),
                    np.full(n_hp, hi),
                    num_restarts,
                    seed,
                ),
                dtype=dt,
            )
        with torch.no_grad():
            _, s1, b1, L1, g1 = _level_objective(correlation, regression, x_lo, self.Y[0], th1)
        self.level1 = _Level(x_lo, th1, 0.0, s1, b1, L1, g1)

        # ---- level 2 (rho, theta2) ----
        x_hi = self.X[1]
        yc = self._level_mean(self.level1, x_hi)

        def obj2(v: Tensor) -> Tensor:
            r, th = v[0], v[1:]
            return _level_objective(correlation, regression, x_hi, self.Y[1] - r * yc, th)[0]

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
                correlation, regression, x_hi, self.Y[1] - rho * yc, th2
            )
        self.level2 = _Level(x_hi, th2, rho, s2, b2, L2, g2)
        self.rho = rho
        self._assemble_joint()

    # ------------------------------------------------------------------ internals
    def _level_mean(self, lvl: _Level, x: Tensor) -> Tensor:
        psi = correlation(self.correlation_name, lvl.theta, x, lvl.X)
        F = regression_matrix(self.regression, x)
        mean = psi @ lvl.gamma
        if F.shape[-1] > 0:
            mean = mean + F @ lvl.beta
        return mean

    def _cross(self, x: Tensor) -> Tensor:
        """``psi(x)`` of shape ``(..., m, n1 + n2)`` (variance-scaled)."""
        l1, l2, rho = self.level1, self.level2, self.rho
        s1, s2 = l1.sigma2, l2.sigma2
        c11 = correlation(self.correlation_name, l1.theta, x, self.X[0])
        c12 = correlation(self.correlation_name, l1.theta, x, self.X[1])
        c22 = correlation(self.correlation_name, l2.theta, x, self.X[1])
        return torch.cat([rho * s1 * c11, rho**2 * s1 * c12 + s2 * c22], dim=-1)

    def _basis(self, x: Tensor) -> Tensor:
        F = regression_matrix(self.regression, x)
        return torch.cat([self.rho * F, F], dim=-1)

    def _assemble_joint(self) -> None:
        l1, l2, rho = self.level1, self.level2, self.rho
        s1, s2 = l1.sigma2, l2.sigma2
        X1, X2 = self.X
        n1, n2 = X1.shape[0], X2.shape[0]
        c = self.correlation_name
        with torch.no_grad():
            S11 = s1 * correlation(c, l1.theta, X1, X1)
            S12 = rho * s1 * correlation(c, l1.theta, X1, X2)
            S22 = rho**2 * s1 * correlation(c, l1.theta, X2, X2) + s2 * correlation(
                c, l2.theta, X2, X2
            )
            S = torch.cat([torch.cat([S11, S12], 1), torch.cat([S12.T, S22], 1)], 0)
            nug = torch.cat(
                [
                    s1 * (n1 + 10) * _EPS * torch.ones(n1, dtype=S.dtype),
                    s2 * (n2 + 10) * _EPS * torch.ones(n2, dtype=S.dtype),
                ]
            )
            S = S + torch.diag(nug)
            F1 = regression_matrix(self.regression, X1)
            F2 = regression_matrix(self.regression, X2)
            p = F1.shape[-1]
            F = torch.cat(
                [
                    torch.cat([F1, F1.new_zeros(n1, p)], 1),
                    torch.cat([rho * F2, F2], 1),
                ],
                0,
            )
            y = torch.cat(self.Y).unsqueeze(-1)
            L = torch.linalg.cholesky(S)
            Yt = _solve_lower(L, y)
            if p > 0:
                Ft = _solve_lower(L, F)
                Q, R = torch.linalg.qr(Ft)
                beta = torch.linalg.solve_triangular(R, Q.T @ Yt, upper=True)
                resid = Yt - Ft @ beta
            else:
                Ft, R, beta, resid = F, None, y.new_zeros(0, 1), Yt
            self._L, self._Ft, self._R = L, Ft, R
            self.beta = beta.squeeze(-1)
            self._gamma = _solve_upper_t(L, resid).squeeze(-1)
            self.sigma2 = rho**2 * s1 + s2  # prior variance of the high-fidelity process

    # ------------------------------------------------------------------ public API
    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def hyperparameters(self) -> dict:
        return {
            "theta1": self.level1.theta.clone(),
            "theta2": self.level2.theta.clone(),
            "rho": self.rho,
            "sigma2_1": float(self.level1.sigma2),
            "sigma2_2": float(self.level2.sigma2),
        }

    def predict_low(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Mean and variance of the low-fidelity (level 1) model alone."""
        x = torch.as_tensor(x, dtype=torch.float64)
        l1 = self.level1
        with torch.no_grad():
            mean = self._level_mean(l1, x)
            psi = correlation(self.correlation_name, l1.theta, x, l1.X)
            F = regression_matrix(self.regression, x)
            Fx = regression_matrix(self.regression, l1.X)
            corrt = _solve_lower(l1.L, psi.T)
            var = 1.0 - (corrt**2).sum(0)
            if F.shape[-1] > 0:
                Ft1 = _solve_lower(l1.L, Fx)
                u = Ft1.T @ corrt - F.T
                var = var + (torch.linalg.solve(Ft1.T @ Ft1, u) * u).sum(0)
            return mean, (l1.sigma2 * var).clamp_min(0.0)

    def _posterior_moments(self, x: Tensor, full_cov: bool) -> tuple[Tensor, Tensor]:
        psi = self._cross(x)  # (..., m, n)
        Fx = self._basis(x)
        mean = psi @ self._gamma
        if Fx.shape[-1] > 0:
            mean = mean + Fx @ self.beta
        corrt = _solve_lower(self._L, psi.transpose(-1, -2))  # (..., n, m)
        c = self.correlation_name
        l1, l2, rho = self.level1, self.level2, self.rho
        if full_cov:
            prior = rho**2 * l1.sigma2 * correlation(c, l1.theta, x, x) + l2.sigma2 * correlation(
                c, l2.theta, x, x
            )
            cov = prior - corrt.transpose(-1, -2) @ corrt
            if Fx.shape[-1] > 0:
                u = self._Ft.T @ corrt - Fx.transpose(-1, -2)
                v = torch.linalg.solve_triangular(self._R.T, u, upper=False)
                cov = cov + v.transpose(-1, -2) @ v
            return mean, cov
        var = self.sigma2 - (corrt**2).sum(-2)
        if Fx.shape[-1] > 0:
            u = self._Ft.T @ corrt - Fx.transpose(-1, -2)
            v = torch.linalg.solve_triangular(self._R.T, u, upper=False)
            var = var + (v**2).sum(-2)
        return mean, var

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """High-fidelity predictive mean and variance, each of shape ``(m,)``."""
        x = torch.as_tensor(x, dtype=torch.float64)
        with torch.no_grad():
            mean, var = self._posterior_moments(x, full_cov=False)
        return mean, var.clamp_min(0.0)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ) -> GPyTorchPosterior:
        """Joint high-fidelity posterior over a ``(..., q, d)`` batch of candidates."""
        X = torch.as_tensor(X, dtype=torch.float64)
        with torch.no_grad():
            mean, cov = self._posterior_moments(X, full_cov=True)
            cov = 0.5 * (cov + cov.transpose(-1, -2))
            cov = cov + 1e-10 * self.sigma2 * torch.eye(cov.shape[-1], dtype=cov.dtype)
        post = GPyTorchPosterior(MultivariateNormal(mean, cov))
        if posterior_transform is not None:
            post = posterior_transform(post)
        return post

    def forward(self, X: Tensor) -> MultivariateNormal:  # pragma: no cover - API completeness
        return self.posterior(X).distribution
