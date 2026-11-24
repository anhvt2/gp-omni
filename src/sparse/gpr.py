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
classification (``likErf`` / ``likLogistic``, Laplace approximation) works with every
combination too; see :class:`~gp_omni.sparse.gpc.GPC`, the classifier twin of this regression model.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gp_omni.settings import load_defaults
from gp_omni.sparse.apx import SparseCov
from gp_omni.sparse.cov import Cov
from gp_omni.sparse.gp import gp_nlz, gp_predict, make_hyp
from gp_omni.sparse.inf import Posterior, inf_gauss_lik, inf_laplace
from gp_omni.sparse.init import init_inducing
from gp_omni.sparse.lik import LikGauss, make_lik
from gp_omni.sparse.mean import Mean
from gp_omni.sparse.multitask import (
    LikGaussTask,
    TaskCov,
    TaskMean,
    augment,
    task_expand,
)
from gp_omni.sparse.train import train

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
    cov, degree : base covariance (``SEard``, ``SEiso``, ``Maternard``, ``Materniso``) when
        ``additive`` is off; ``degree`` is the Matern degree.
    rank : rank of the output coregionalisation ``B = W W' + diag(kappa^2)``.
    lik : ``likGauss`` (regression), or ``likErf`` / ``likLogistic`` (labels +/-1); any
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
        make_lik(lik)  # fail early on an unknown likelihood name
        self.hyp: dict[str, Tensor] | None = None
        self.post: Posterior | None = None
        self.nlZ: float | None = None

    # ------------------------------------------------------------------ construction
    def _build(self, D: int) -> None:
        base = (
            Cov("ADD", orders=self.additive) if self.additive else Cov(self.cov_kind, self.degree)
        )
        self.base = base
        gauss = self.lik_name == "likGauss"
        if self.multitask:
            self.cov = TaskCov(base, D, self.O, self.L, self.rank)
            self.mean = TaskMean(D, self.O, self.L)
            self.lik = LikGaussTask(D, self.O, self.L) if gauss else make_lik(self.lik_name)
        else:
            self.cov, self.mean, self.lik = base, Mean("const"), make_lik(self.lik_name)
        self.inf = inf_gauss_lik if isinstance(self.lik, LikGauss) else inf_laplace

    def _base_hyp(self, x: Tensor, sf: float) -> Tensor:
        D = x.shape[1]
        ell = torch.log(x.std(0).clamp_min(DEFAULTS["initialization"]["lengthscale_floor"]))
        if self.base.kind == "ADD":
            amp = torch.tensor([-0.5 * math.log(math.comb(D, r)) for r in self.base.orders])
            return torch.cat([ell, amp])
        if self.base.kind.endswith("ard"):
            return torch.cat([ell, torch.tensor([math.log(sf)])])
        return torch.stack([ell.mean(), torch.tensor(math.log(sf))])

    def _expand(self):
        return task_expand(self._D, self.O, self.L) if self.multitask else None

    def _sparse_cov(self, xu: Tensor | None) -> SparseCov | TaskCov | Cov:
        if self.nu is None:
            return self.cov
        return SparseCov(self.cov, xu, expand=self._expand())

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
        """Fit on inputs ``x`` (n, D), targets ``y`` (n,), and, for multi-output /
        multi-fidelity models, the integer ``output`` / ``fidelity`` index of every row."""
        x, y = x.to(torch.float64), y.to(torch.float64).reshape(-1)
        n, D = x.shape
        self._D = D
        if y.shape[0] != n:
            raise ValueError("x and y must have the same number of rows")
        o = torch.zeros(n, dtype=torch.long) if output is None else torch.as_tensor(output).long()
        f = (
            torch.zeros(n, dtype=torch.long)
            if fidelity is None
            else torch.as_tensor(fidelity).long()
        )
        if int(o.max()) >= self.O or int(f.max()) >= self.L or int(o.min()) < 0 or int(f.min()) < 0:
            raise ValueError(
                "output / fidelity indices must lie in [0, num_outputs) / [0, num_fidelities)"
            )
        self._build(D)
        gauss = isinstance(self.lik, LikGauss)
        if not gauss and not set(y.unique().tolist()) <= {-1.0, 1.0}:
            raise ValueError("classification labels must be in {-1, +1}")

        # per-task standardisation (Gaussian likelihoods only)
        T = self.O * self.L
        self._mu, self._sd = torch.zeros(T), torch.ones(T)
        t = o * self.L + f
        ys = y.clone()
        if gauss:
            for k in range(T):
                m = t == k
                if m.sum() >= 1:
                    self._mu[k] = y[m].mean()
                if m.sum() >= 2 and y[m].std() > 0:
                    self._sd[k] = y[m].std()
            ys = (y - self._mu[t]) / self._sd[t]

        sf = DEFAULTS["initialization"][
            "regression_signal_std" if gauss else "classification_signal_std"
        ]
        bh = self._base_hyp(x, sf)
        if self.multitask:
            hyp = make_hyp(
                mean=torch.zeros(T),
                cov=self.cov.initial_hyp(bh),
                lik=torch.full((T,), math.log(DEFAULTS["initialization"]["noise_std"]))
                if gauss
                else torch.zeros(0),
            )
        else:
            hyp = make_hyp(
                mean=torch.zeros(1),
                cov=bh,
                lik=torch.full((1,), math.log(DEFAULTS["initialization"]["noise_std"]))
                if gauss
                else torch.zeros(0),
            )
        xu0 = None
        if self.nu is not None:
            xu0 = init_inducing(x, self.nu, self.init, self.seed)
            if learn_xu:
                hyp["xu"] = xu0.clone()
        self._xu0 = xu0
        self._xa = augment(x, o, f) if self.multitask else x
        self._ys = ys
        cov = self._sparse_cov(xu0)
        opt = {"s": self.s}
        hyp, self.nlZ = train(hyp, self.inf, self.mean, cov, self.lik, self._xa, ys, opt, max_iter)
        self.hyp, self._opt = hyp, opt
        _, _, self.post = gp_nlz(
            hyp, self.inf, self.mean, cov, self.lik, self._xa, ys, opt, grad=False
        )
        return self

    # --------------------------------------------------------------------- predict
    def _predict_one(self, xs: Tensor, o: int, f: int) -> dict[str, Tensor]:
        xa = augment(xs, torch.full((xs.shape[0],), float(o)), torch.full((xs.shape[0],), float(f)))
        xa = xa if self.multitask else xs
        out = gp_predict(
            self.hyp, self.post, self.mean, self._sparse_cov(self._xu0), self.lik, self._xa, xa
        )
        if not isinstance(self.lik, LikGauss):  # classification: p(y = +1 | data)
            out["prob"] = torch.clamp((out["ymu"] + 1) / 2, 0.0, 1.0)
        if isinstance(self.lik, LikGauss):  # undo the per-task standardisation
            k = o * self.L + f
            sd, mu = self._sd[k], self._mu[k]
            out = {
                "fmu": out["fmu"] * sd + mu,
                "ymu": out["ymu"] * sd + mu,
                "fs2": out["fs2"] * sd**2,
                "ys2": out["ys2"] * sd**2,
            }
        return out

    def predict(
        self, xs: Tensor, output: int | None = None, fidelity: int | None = None
    ) -> dict[str, Tensor]:
        """Predictive ``fmu, fs2`` (latent) and ``ymu, ys2`` (with noise / class probability).

        ``fidelity`` defaults to the highest. With several outputs and ``output=None`` every
        entry has shape ``(m, num_outputs)``.
        """
        if self.hyp is None or self.post is None:
            raise RuntimeError("call fit() first")
        xs = xs.to(torch.float64)
        f = self.L - 1 if fidelity is None else fidelity
        if output is not None or self.O == 1:
            return self._predict_one(xs, output or 0, f)
        per = [self._predict_one(xs, k, f) for k in range(self.O)]
        return {key: torch.stack([p[key] for p in per], dim=1) for key in per[0]}

    @property
    def inducing_inputs(self) -> Tensor | None:
        if self.hyp is None:
            raise RuntimeError("call fit() first")
        if self.nu is None:
            return None
        return self.hyp["xu"].detach() if "xu" in self.hyp else self._xu0
