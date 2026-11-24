"""Exact co-kriging: legacy two-fidelity AR1 and general hierarchical fidelity DAGs.

The AR1 implementation ports the owner's MATLAB ``@CoKriging``. Its model is

    f_lo(x) ~ GP(F(x) b1, s1 * Psi1(x, x'))
    f_hi(x) = rho * f_lo(x) + delta(x),   delta ~ GP(F(x) b2, s2 * Psi2(x, x'))

Fitting is sequential, as in ``@CoKriging/fit.m``:

1. fit level 1 on the low-fidelity data (concentrated likelihood, ``s1`` profiled out);
2. fit level 2 on ``y_hi - rho * mu_lo(X_hi)`` over ``(rho, theta2)``;
3. assemble the *joint* covariance of ``[y_lo; y_hi]`` and compute a generalised
   least-squares mean (``b = [b1; b2]``), used for prediction.

Differences from the MATLAB code, deliberately:

* exactly two fidelity levels (same limit as the MATLAB code);
* the predictive-variance regression term uses ``(Ft' Ft)^-1 = R^-1 R^-T``; the MATLAB code solves
  with ``R`` instead of ``R'``, which only differs when the regression basis has > 1 column;
* hyperparameters are fitted with autograd + L-BFGS-B multi-start instead of SQPLab.

Inputs are used as given (no normalisation), like the MATLAB code. Scale them to a comparable range
first (e.g. the unit cube), since the default bounds assume that.

``DAGCoKriging`` extends the autoregressive model to arbitrary acyclic graphs,
with independent per-node discrepancies and joint likelihood fitting. The legacy
``AR1CoKriging`` retains its sequential fitting and MATLAB compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from scipy.optimize import minimize
from torch import Tensor

from gp_omni.correlation import CORRELATIONS, correlation
from gp_omni.regression import regression_matrix
from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("models")

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
        ``"none"`` (zero-mean, the MATLAB default), ``"regpoly0"`` (constant),
        ``"regpoly1"`` or ``"regpoly2"``. Shared by both levels.
    hp0, hp_bounds
        Initial ``log10`` precision (scalar or per-dim) and ``(lo, hi)`` bounds.
    rho0, rho_bounds
        Initial scaling factor and its bounds (MATLAB defaults: 5 and ``[0.1, 5]``).
    hyperparameters
        Optional ``{"theta1", "theta2", "rho"}``; if given, no optimisation is run.
    """

    def __init__(
        self,
        train_X: list[Tensor],
        train_Y: list[Tensor],
        correlation: str = DEFAULTS["ar1"]["correlation"],
        regression: str = DEFAULTS["ar1"]["regression"],
        hp0: float | list[float] = DEFAULTS["ar1"]["hp0"],
        hp_bounds: tuple[float, float] = tuple(DEFAULTS["ar1"]["hp_bounds"]),
        rho0: float = DEFAULTS["ar1"]["rho0"],
        rho_bounds: tuple[float, float] = tuple(DEFAULTS["ar1"]["rho_bounds"]),
        num_restarts: int = DEFAULTS["ar1"]["num_restarts"],
        seed: int = DEFAULTS["ar1"]["seed"],
        hyperparameters: dict | None = DEFAULTS["ar1"]["hyperparameters"],
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


class DAGCoKriging(Model):
    r"""Exact, noiseless co-kriging on a supplied directed acyclic graph.

    Node ``j`` is ``sum(rho[i,j] * f_i for i in parents(j)) + delta_j``,
    with independent discrepancy GPs. Missing edges fix direct weights to zero;
    indirect paths and common ancestors can still induce covariance.

    ``train_X`` and ``train_Y`` are lists indexed by node, with nonnested designs
    allowed. ``edges`` contains directed ``(parent, child)`` pairs in any order.
    A unique sink is the default prediction target; otherwise ``target`` is required.
    All nodes share the correlation family and regression degree, with independent
    per-node precisions and variances. Joint maximum likelihood profiles out trends.

    ``hyperparameters`` skips fitting and requires ``theta`` of shape ``(L, d)``
    (or ``(L, 1)`` for isotropic kernels), ``rho`` of shape ``(E,)`` in supplied
    edge order, and positive ``sigma2`` of shape ``(L,)``. Unlike the legacy model,
    fixed variances are honored. Inputs should be scaled, e.g. to the unit cube.
    Fit uses CPU float64; the fitted model supports device/dtype conversion with
    ``to``. Dense fitting costs O(N^3) time and O(N^2) memory.
    """

    def __init__(
        self,
        train_X: list[Tensor],
        train_Y: list[Tensor],
        edges: list[tuple[int, int]],
        target: int | None = DEFAULTS["dag"]["target"],
        correlation: str = DEFAULTS["dag"]["correlation"],
        regression: str = DEFAULTS["dag"]["regression"],
        hp0: float | list[float] = DEFAULTS["dag"]["hp0"],
        hp_bounds: tuple[float, float] = tuple(DEFAULTS["dag"]["hp_bounds"]),
        rho0: float = DEFAULTS["dag"]["rho0"],
        rho_bounds: tuple[float, float] = tuple(DEFAULTS["dag"]["rho_bounds"]),
        log_variance_bounds: tuple[float, float] = tuple(DEFAULTS["dag"]["log_variance_bounds"]),
        num_restarts: int = DEFAULTS["dag"]["num_restarts"],
        seed: int = DEFAULTS["dag"]["seed"],
        hyperparameters: dict | None = DEFAULTS["dag"]["hyperparameters"],
    ) -> None:
        super().__init__()
        self.num_nodes = len(train_X)
        if not self.num_nodes or len(train_Y) != self.num_nodes:
            raise ValueError("need matching nonempty lists of node inputs and outputs")
        if correlation not in CORRELATIONS:
            raise ValueError(f"unknown correlation {correlation!r}")
        self.correlation_name, self.regression = correlation, regression
        self.edges = tuple(tuple(e) for e in edges)
        parents = [[] for _ in train_X]
        children = [[] for _ in train_X]
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("duplicate edges are not allowed")
        for e in self.edges:
            if len(e) != 2 or any(
                not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < self.num_nodes
                for i in e
            ):
                raise ValueError("edges must contain valid integer (parent, child) pairs")
            i, j = e
            parents[j].append(i)
            children[i].append(j)
        pending = [len(p) for p in parents]
        order = [i for i, count in enumerate(pending) if count == 0]
        for i in order:
            for j in children[i]:
                pending[j] -= 1
                if pending[j] == 0:
                    order.append(j)
        if len(order) != self.num_nodes:
            raise ValueError("graph must be acyclic (self-edges are also forbidden)")
        self.topological_order = tuple(order)
        sinks = [i for i, c in enumerate(children) if not c]
        if target is None:
            if len(sinks) != 1:
                raise ValueError("target is required when the graph has multiple sinks")
            target = sinks[0]
        self.target = self._node(target)
        for i, (x, y) in enumerate(zip(train_X, train_Y)):
            x = torch.as_tensor(x, dtype=torch.float64, device="cpu").detach().clone()
            y = torch.as_tensor(y, dtype=torch.float64, device="cpu").detach().clone()
            if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
                raise ValueError("each X must be a nonempty (n, d) matrix")
            if y.shape not in ((x.shape[0],), (x.shape[0], 1)):
                raise ValueError("each Y must have shape (n,) or (n, 1)")
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError("training data must be finite")
            if i == 0:
                self.dim = x.shape[1]
            elif x.shape[1] != self.dim:
                raise ValueError("all nodes must share the input dimension")
            F = regression_matrix(regression, x)
            if F.shape[1] and (
                x.shape[0] <= F.shape[1] or torch.linalg.matrix_rank(F) < F.shape[1]
            ):
                raise ValueError(
                    "each node needs a full-rank trend and residual degrees of freedom"
                )
            self.register_buffer(f"_X{i}", x)
            self.register_buffer(f"_Y{i}", y.reshape(-1))

        size = self.num_nodes * self.dim
        if hyperparameters is not None:
            theta = torch.as_tensor(hyperparameters["theta"], dtype=torch.float64, device="cpu")
            if theta.shape == (self.num_nodes, 1):
                theta = theta.expand(-1, self.dim).clone()
            rho = torch.as_tensor(hyperparameters["rho"], dtype=torch.float64, device="cpu")
            variances = torch.as_tensor(
                hyperparameters["sigma2"], dtype=torch.float64, device="cpu"
            )
        else:
            bounds = (hp_bounds, rho_bounds, log_variance_bounds)
            if any(len(b) != 2 or not np.isfinite(b).all() or b[0] >= b[1] for b in bounds):
                raise ValueError("hyperparameter bounds must be finite increasing pairs")
            if not isinstance(num_restarts, Integral) or num_restarts < 0:
                raise ValueError("num_restarts must be a nonnegative integer")
            h0 = np.broadcast_to(np.asarray(hp0, dtype=float), (self.dim,)).copy()
            if not np.isfinite(h0).all() or not np.isfinite(rho0):
                raise ValueError("initial hyperparameters must be finite")
            h0 = np.clip(h0, *hp_bounds)
            r0 = float(np.clip(rho0, *rho_bounds))
            # Sequential parent means supply initial scales only. Fitting below is joint.
            initial_levels = {}
            A0 = self._loadings(torch.full((len(self.edges),), r0, dtype=torch.float64))
            scales = torch.zeros(self.num_nodes, dtype=torch.float64)
            for j in order:
                residual = self.Y[j].clone()
                for q in order[: order.index(j)]:
                    if A0[j, q] != 0:
                        lvl = initial_levels[q]
                        mu = self._discrepancy_mean(lvl, self.X[j])
                        residual = residual - A0[j, q] * mu
                th = torch.tensor(h0)
                with torch.no_grad():
                    _, s, b, L, g = _level_objective(
                        correlation, regression, self.X[j], residual, th
                    )
                scales[j] = s.clamp_min(max(float(self.Y[j].square().mean()), 1.0) * 1e-10)
                initial_levels[j] = _Level(self.X[j], th, r0, scales[j], b, L, g)

            def objective(v: Tensor) -> Tensor:
                th = v[:size].reshape(self.num_nodes, self.dim)
                r = v[size : size + len(self.edges)]
                s = scales * torch.pow(10.0, v[size + len(self.edges) :])
                L, _, _, _, resid = self._factor(th, r, s)
                return 0.5 * resid.square().sum() + torch.log(L.diagonal()).sum()

            x0 = np.concatenate(
                [
                    np.tile(h0, self.num_nodes),
                    np.full(len(self.edges), r0),
                    np.zeros(self.num_nodes),
                ]
            )
            lower = np.concatenate(
                [
                    np.full(size, hp_bounds[0]),
                    np.full(len(self.edges), rho_bounds[0]),
                    np.full(self.num_nodes, log_variance_bounds[0]),
                ]
            )
            upper = np.concatenate(
                [
                    np.full(size, hp_bounds[1]),
                    np.full(len(self.edges), rho_bounds[1]),
                    np.full(self.num_nodes, log_variance_bounds[1]),
                ]
            )
            sol = torch.tensor(_minimize(objective, x0, lower, upper, num_restarts, seed))
            theta = sol[:size].reshape(self.num_nodes, self.dim)
            rho = sol[size : size + len(self.edges)]
            variances = scales * torch.pow(10.0, sol[size + len(self.edges) :])
        if (
            theta.shape != (self.num_nodes, self.dim)
            or rho.shape != (len(self.edges),)
            or variances.shape != (self.num_nodes,)
        ):
            raise ValueError("invalid theta, rho, or sigma2 shape")
        if (
            not all(torch.isfinite(t).all() for t in (theta, rho, variances))
            or not (variances > 0).all()
        ):
            raise ValueError("hyperparameters must be finite and sigma2 strictly positive")
        self.register_buffer("theta", theta.detach().clone())
        self.register_buffer("rho", rho.detach().clone())
        self.register_buffer("variances", variances.detach().clone())
        with torch.no_grad():
            L, Ft, R, beta, resid = self._factor(self.theta, self.rho, self.variances)
            self.register_buffer("loadings", self._loadings(self.rho))
            self.register_buffer("_L", L)
            self.register_buffer("_Ft", Ft)
            self.register_buffer("_R", R)
            self.register_buffer("beta", beta.squeeze(-1))
            self.register_buffer("_gamma", _solve_upper_t(L, resid).squeeze(-1))

    @property
    def X(self) -> list[Tensor]:
        return [getattr(self, f"_X{i}") for i in range(self.num_nodes)]

    @property
    def Y(self) -> list[Tensor]:
        return [getattr(self, f"_Y{i}") for i in range(self.num_nodes)]

    def _node(self, node: int) -> int:
        if (
            not isinstance(node, Integral)
            or isinstance(node, bool)
            or not 0 <= node < self.num_nodes
        ):
            raise ValueError("fidelity must be a valid integer node index")
        return int(node)

    def _loadings(self, rho: Tensor) -> Tensor:
        # A[j,q] sums the products of edge weights over every path q -> j.
        rows = {}
        eye = torch.eye(self.num_nodes, dtype=rho.dtype, device=rho.device)
        for j in self.topological_order:
            row = eye[j]
            for e, (i, child) in enumerate(self.edges):
                if child == j:
                    row = row + rho[e] * rows[i]
            rows[j] = row
        return torch.stack([rows[j] for j in range(self.num_nodes)])

    def _discrepancy_mean(self, level: _Level, x: Tensor) -> Tensor:
        return (
            correlation(self.correlation_name, level.theta, x, level.X) @ level.gamma
            + regression_matrix(self.regression, x) @ level.beta
        )

    def _covariance(self, i, j, x, z, A, theta, variances):
        return sum(
            A[i, q] * A[j, q] * variances[q] * correlation(self.correlation_name, theta[q], x, z)
            for q in range(self.num_nodes)
        )

    def _design(self, node: int, x: Tensor, A: Tensor) -> Tensor:
        F = regression_matrix(self.regression, x)
        return torch.cat([A[node, q] * F for q in range(self.num_nodes)], dim=-1)

    def _factor(self, theta, rho, variances):
        A = self._loadings(rho)
        S = torch.cat(
            [
                torch.cat(
                    [
                        self._covariance(i, j, x, z, A, theta, variances)
                        for j, z in enumerate(self.X)
                    ],
                    dim=-1,
                )
                for i, x in enumerate(self.X)
            ],
            dim=-2,
        )
        # Relative numerical jitter, not observation noise.
        S = S + torch.diag(S.diagonal() * (S.shape[0] + 10) * _EPS)
        F = torch.cat([self._design(i, x, A) for i, x in enumerate(self.X)])
        L = torch.linalg.cholesky(S)
        Yt = _solve_lower(L, torch.cat(self.Y).unsqueeze(-1))
        Ft = _solve_lower(L, F)
        if F.shape[-1]:
            Q, R = torch.linalg.qr(Ft)
            beta = torch.linalg.solve_triangular(R, Q.T @ Yt, upper=True)
        else:
            R, beta = F.new_zeros(0, 0), F.new_zeros(0, 1)
        return L, Ft, R, beta, Yt - Ft @ beta

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def hyperparameters(self) -> dict:
        return {
            "theta": self.theta.clone(),
            "rho": self.rho.clone(),
            "sigma2": self.variances.clone(),
        }

    @property
    def coupling_matrix(self) -> Tensor:
        """Direct weights B[parent, child]; absent edges are exactly zero."""
        B = self.rho.new_zeros(self.num_nodes, self.num_nodes)
        for weight, (i, j) in zip(self.rho, self.edges):
            B[i, j] = weight
        return B

    def _moments(self, x: Tensor, node: int, full_cov: bool):
        A = self.loadings
        cross = torch.cat(
            [
                self._covariance(node, j, x, z, A, self.theta, self.variances)
                for j, z in enumerate(self.X)
            ],
            dim=-1,
        )
        F = self._design(node, x, A)
        mean = cross @ self._gamma + F @ self.beta
        v = _solve_lower(self._L, cross.transpose(-1, -2))
        prior_var = (A[node].square() * self.variances).sum()
        if full_cov:
            cov = self._covariance(node, node, x, x, A, self.theta, self.variances)
            cov = cov - v.transpose(-1, -2) @ v
        else:
            cov = prior_var - v.square().sum(-2)
        if F.shape[-1]:
            u = self._Ft.T @ v - F.transpose(-1, -2)
            t = torch.linalg.solve_triangular(self._R.T, u, upper=False)
            cov = cov + (t.transpose(-1, -2) @ t if full_cov else t.square().sum(-2))
        return mean, cov

    def _inputs(self, x: Tensor) -> Tensor:
        x = torch.as_tensor(x, dtype=self.theta.dtype, device=self.theta.device)
        if x.ndim < 2 or x.shape[-1] != self.dim or not torch.isfinite(x).all():
            raise ValueError("prediction inputs must be finite with shape (..., q, d)")
        return x

    def predict(self, x: Tensor, fidelity: int | None = None) -> tuple[Tensor, Tensor]:
        """Mean and marginal variance at a node, conditioned on all training data."""
        node = self.target if fidelity is None else self._node(fidelity)
        mean, var = self._moments(self._inputs(x), node, full_cov=False)
        return mean, var.clamp_min(0)

    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
        fidelity: int | None = None,
    ) -> GPyTorchPosterior:
        """Differentiable joint posterior for one selected node; supports t-batches."""
        if output_indices is not None and output_indices != [0]:
            raise ValueError("the selected fidelity has one output, indexed by 0")
        if not isinstance(observation_noise, bool) or observation_noise:
            raise NotImplementedError("this noiseless model only supports observation_noise=False")
        node = self.target if fidelity is None else self._node(fidelity)
        mean, cov = self._moments(self._inputs(X), node, full_cov=True)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        scale = (self.loadings[node].square() * self.variances).sum()
        cov = cov + 1e-10 * scale * torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
        post = GPyTorchPosterior(MultivariateNormal(mean, cov))
        return posterior_transform(post) if posterior_transform is not None else post

    def forward(self, X: Tensor) -> MultivariateNormal:
        return self.posterior(X).distribution
