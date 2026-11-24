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

``dag_cokriging`` extends the autoregressive model to arbitrary acyclic graphs,
with independent per-node discrepancies and joint likelihood fitting. The legacy
``ar1_cokriging`` retains its sequential fitting and MATLAB compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
from botorch.models.model import Model
from torch import Tensor

from gp_omni.build.correlation import CORRELATIONS, correlation
from gp_omni.build.regression import regression_matrix
from gp_omni.infer.cokriging import ar1_prediction, dag_prediction
from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("models")


@dataclass
class _level:
    """One fitted universal-kriging level (MATLAB ``BasicGaussianProcess`` without noise)."""

    X: Tensor
    theta: Tensor  # log10 precisions
    rho: float
    sigma2: Tensor
    beta: Tensor
    L: Tensor
    gamma: Tensor  # Psi^-1 (y - F beta)


class ar1_cokriging(ar1_prediction, Model):
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
            raise ValueError("ar1_cokriging supports exactly two fidelity levels [low, high]")
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

        self._fitted = False
        self._fit_options = {
            "hp0": hp0,
            "hp_bounds": hp_bounds,
            "rho0": rho0,
            "rho_bounds": rho_bounds,
            "num_restarts": num_restarts,
            "seed": seed,
            "hyperparameters": hyperparameters,
        }

    def fit(self, **overrides):
        """Fit the prepared problem; construction never optimizes hyperparameters."""
        from gp_omni.train.cokriging import fit_ar1

        return fit_ar1(self, **{**self._fit_options, **overrides})

    # ------------------------------------------------------------------ public API
    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def hyperparameters(self) -> dict:
        self._require_fitted()
        return {
            "theta1": self.level1.theta.clone(),
            "theta2": self.level2.theta.clone(),
            "rho": self.rho,
            "sigma2_1": float(self.level1.sigma2),
            "sigma2_2": float(self.level2.sigma2),
        }


class dag_cokriging(dag_prediction, Model):
    r"""Exact, noiseless co-kriging on a supplied directed acyclic graph.

    Node ``j`` is ``sum(rho[i,j] * f_i for i in parents(j)) + delta_j``,
    with independent discrepancy GPs. Missing edges fix direct weights to zero;
    indirect paths and common ancestors can still induce covariance.

    ``train_X`` and ``train_Y`` are lists indexed by node, with nonnested designs
    allowed. ``edges`` contains directed ``(parent, child)`` pairs in any order.
    A unique sink is the default prediction target; otherwise ``target`` is required.
    All nodes share the correlation family and regression degree, with independent
    per-node precisions and variances. Joint maximum likelihood profiles out trends.

    Call ``fit()`` after construction. ``hyperparameters`` skips optimization and requires ``theta`` of shape ``(L, d)``
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

        self._fitted = False
        self._fit_options = {
            "hp0": hp0,
            "hp_bounds": hp_bounds,
            "rho0": rho0,
            "rho_bounds": rho_bounds,
            "log_variance_bounds": log_variance_bounds,
            "num_restarts": num_restarts,
            "seed": seed,
            "hyperparameters": hyperparameters,
        }

    def fit(self, **overrides):
        """Fit the prepared problem; construction never optimizes hyperparameters."""
        from gp_omni.train.cokriging import fit_dag

        return fit_dag(self, **{**self._fit_options, **overrides})

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

    def _covariance(self, i, j, x, z, A, theta, variances):
        return sum(
            A[i, q] * A[j, q] * variances[q] * correlation(self.correlation_name, theta[q], x, z)
            for q in range(self.num_nodes)
        )

    def _design(self, node: int, x: Tensor, A: Tensor) -> Tensor:
        F = regression_matrix(self.regression, x)
        return torch.cat([A[node, q] * F for q in range(self.num_nodes)], dim=-1)

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def hyperparameters(self) -> dict:
        self._require_fitted()
        return {
            "theta": self.theta.clone(),
            "rho": self.rho.clone(),
            "sigma2": self.variances.clone(),
        }

    @property
    def coupling_matrix(self) -> Tensor:
        """Direct weights B[parent, child]; absent edges are exactly zero."""
        self._require_fitted()
        B = self.rho.new_zeros(self.num_nodes, self.num_nodes)
        for weight, (i, j) in zip(self.rho, self.edges):
            B[i, j] = weight
        return B
