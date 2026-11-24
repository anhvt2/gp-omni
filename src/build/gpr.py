"""``GPR``: one GP with four independent switches.

=====================  ===========================  ===================================
feature                switch                       what it does
=====================  ===========================  ===================================
correlated outputs     ``num_outputs > 1``          coregionalised (ICM/LMC) outputs
multi-fidelity         ``num_fidelities > 1``       autoregressive fidelities (AR1)
big data / sparse      ``nu = <int>``               gpml sparse VFE / FITC / SPEP
additive (high-dim)    ``additive = R`` or list     gpml ``covADD`` of orders ``1..R``
=====================  ===========================  ===================================

Every combination is valid. Each switch changes one ingredient only (the covariance, or
how the covariance is approximated), so the features compose: e.g. all four at once give a
sparse additive GP over several correlated outputs at several fidelities.

Targets are standardised per task, ``(output, fidelity)``, before fitting and predictions are
mapped back. Regression uses Gaussian noise (one level per task);
classification (``probit`` / ``logistic``, Laplace approximation) works with every
combination too; see :class:`~gp_omni.build.gpc.GPC`, the classifier twin of this regression model.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Self

import torch
from torch import Tensor

from gp_omni.build.covariance import covariance
from gp_omni.build.likelihood import gaussian, make_likelihood
from gp_omni.build.mean import mean_function
from gp_omni.build.multitask import (
    task_covariance,
    task_expand,
    task_gaussian,
    task_mean,
)
from gp_omni.build.sparse_covariance import sparse_covariance
from gp_omni.infer.posterior import gaussian_posterior, laplace_posterior, posterior_state
from gp_omni.settings import load_defaults, load_model_config

DEFAULTS = load_defaults("models")


class GPR:
    """Gaussian process with independent multi-output, multi-fidelity, sparse and additive
    switches (see the module docstring).

    Parameters
    ----------
    num_outputs, num_fidelities : sizes of the output and fidelity axes (default 1 = off).
    additive : ``None`` (off), an int ``R`` (orders ``1..R``) or an explicit list of orders.
    nu : number of inducing locations; ``None`` means exact (dense) inference.
    s : sparse power in ``[0, 1]``; ``0`` = variational free energy (default), ``1`` = FITC.
    cov, degree : base covariance (``se_ard``, ``se_iso``, ``matern_ard``, ``matern_iso``) when
        ``additive`` is off; ``degree`` is the Matern degree.
    rank : rank of the output coregionalisation ``B = W W' + diag(kappa^2)``.
    lik : ``gaussian`` (regression), or ``probit`` / ``logistic`` (labels +/-1); any
        combination of the four features.
    init, seed : inducing-input initialisation (``kmeans`` or ``random``) and its seed.
    """

    def __init__(
        self,
        num_outputs: int = DEFAULTS["gpr"]["num_outputs"],
        num_fidelities: int = DEFAULTS["gpr"]["num_fidelities"],
        additive: int | list[int] | None = DEFAULTS["gpr"]["additive"],
        nu: int | None = DEFAULTS["gpr"]["nu"],
        s: float = DEFAULTS["gpr"]["s"],
        cov: str = DEFAULTS["gpr"]["cov"],
        degree: int = DEFAULTS["gpr"]["degree"],
        rank: int = DEFAULTS["gpr"]["rank"],
        lik: str = DEFAULTS["gpr"]["lik"],
        init: str = DEFAULTS["gpr"]["init"],
        seed: int = DEFAULTS["gpr"]["seed"],
    ):
        self.O, self.L, self.nu, self.s = num_outputs, num_fidelities, nu, s
        self.additive, self.cov_kind, self.degree, self.rank = additive, cov, degree, rank
        self.init, self.seed = init, seed
        self.multitask = num_outputs * num_fidelities > 1
        self.lik_name = lik
        make_likelihood(lik)  # fail early on an unknown likelihood name
        self.hyp: dict[str, Tensor] | None = None
        self.post: posterior_state | None = None
        self.nlZ: float | None = None

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> Self:
        """Construct from a flat JSON settings file, with optional keyword overrides."""
        return cls(**load_model_config(path, **overrides))

    # ------------------------------------------------------------------ construction
    def _build(self, D: int) -> None:
        base = (
            covariance("additive", orders=self.additive)
            if self.additive
            else covariance(self.cov_kind, self.degree)
        )
        self.base = base
        gauss = self.lik_name == "gaussian"
        if self.multitask:
            self.cov = task_covariance(base, D, self.O, self.L, self.rank)
            self.mean = task_mean(D, self.O, self.L)
            self.lik = task_gaussian(D, self.O, self.L) if gauss else make_likelihood(self.lik_name)
        else:
            self.cov, self.mean, self.lik = (
                base,
                mean_function("const"),
                make_likelihood(self.lik_name),
            )
        self.inf = gaussian_posterior if isinstance(self.lik, gaussian) else laplace_posterior

    def _base_hyp(self, x: Tensor, sf: float) -> Tensor:
        D = x.shape[1]
        ell = torch.log(x.std(0).clamp_min(DEFAULTS["initialization"]["lengthscale_floor"]))
        if self.base.kind == "additive":
            amp = torch.tensor([-0.5 * math.log(math.comb(D, r)) for r in self.base.orders])
            return torch.cat([ell, amp])
        if self.base.kind.endswith("ard"):
            return torch.cat([ell, torch.tensor([math.log(sf)])])
        return torch.stack([ell.mean(), torch.tensor(math.log(sf))])

    def _expand(self):
        return task_expand(self._D, self.O, self.L) if self.multitask else None

    def _sparse_cov(self, xu: Tensor | None) -> sparse_covariance | task_covariance | covariance:
        if self.nu is None:
            return self.cov
        return sparse_covariance(self.cov, xu, expand=self._expand())

    # ------------------------------------------------------------------------- fit
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        output: Tensor | None = None,
        fidelity: Tensor | None = None,
        max_iter: int = DEFAULTS["gpr_fit"]["max_iter"],
        learn_xu: bool = DEFAULTS["gpr_fit"]["learn_xu"],
    ) -> GPR:
        """Fit hyperparameters through :mod:`gp_omni.train.gpr`."""
        from gp_omni.train.gpr import fit

        return fit(self, x, y, output, fidelity, max_iter, learn_xu)

    # --------------------------------------------------------------------- predict

    def predict(
        self, xs: Tensor, output: int | None = None, fidelity: int | None = None
    ) -> dict[str, Tensor]:
        """Predict from the fitted model through :mod:`gp_omni.infer.gpr`."""
        from gp_omni.infer.gpr import predict

        return predict(self, xs, output, fidelity)

    @property
    def inducing_inputs(self) -> Tensor | None:
        if self.hyp is None:
            raise RuntimeError("call fit() first")
        if self.nu is None:
            return None
        return self.hyp["xu"].detach() if "xu" in self.hyp else self._xu0
