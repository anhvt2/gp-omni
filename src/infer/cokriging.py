"""Co-kriging predictive moments and BoTorch posterior adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor

from gp_omni.build.correlation import correlation
from gp_omni.build.linalg import _solve_lower
from gp_omni.build.regression import regression_matrix

if TYPE_CHECKING:
    from gp_omni.build.cokriging import _Level


class ar1_prediction:
    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("call fit() first")

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

    def predict_low(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """mean_function and variance of the low-fidelity (level 1) model alone."""
        self._require_fitted()
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
        self._require_fitted()
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
        self._require_fitted()
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


class dag_prediction:
    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("call fit() first")

    def _discrepancy_mean(self, level: _Level, x: Tensor) -> Tensor:
        return (
            correlation(self.correlation_name, level.theta, x, level.X) @ level.gamma
            + regression_matrix(self.regression, x) @ level.beta
        )

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
        """mean_function and marginal variance at a node, conditioned on all training data."""
        self._require_fitted()
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
        self._require_fitted()
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
