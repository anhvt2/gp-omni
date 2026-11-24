"""End-to-end sparse variational GP: initialise, (optionally) swap, train, predict.

The default ``s = 0`` is the variational free energy bound (Titsias 2009); ``s = 1`` gives
FITC and ``0 < s < 1`` sparse power EP.
Gaussian likelihoods use ``infGaussLik``; ``probit`` / ``logistic`` (labels +/-1) use
``infLaplace`` on the same sparse covariance approximation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Self

import torch
from torch import Tensor

from gp_omni.build.covariance import covariance
from gp_omni.build.hyperparameters import make_hyperparameters
from gp_omni.build.inducing import init_inducing
from gp_omni.build.likelihood import gaussian, make_likelihood
from gp_omni.build.mean import mean_function
from gp_omni.build.sparse_covariance import sparse_covariance
from gp_omni.infer.posterior import gaussian_posterior, laplace_posterior, posterior_state
from gp_omni.settings import load_defaults, load_model_config

DEFAULTS = load_defaults("models")
INFERENCE = load_defaults("inference")


class sparse_gp:
    """Sparse GP regression / classification.

    Parameters
    ----------
    nu : number of inducing inputs.
    cov, degree : ``"se_ard"``, ``"se_iso"``, ``"matern_ard"``, ``"matern_iso"`` (+ Matern degree).
    mean : ``"zero"``, ``"const"``, ``"linear"`` or ``"linear+const"``.
    lik : ``"gaussian"``, ``"probit"`` or ``"logistic"``.
    s : sparse power in ``[0, 1]``: ``0`` VFE (default), ``1`` FITC.
    init : ``"kmeans"`` (Improved-Nystrom landmarks, randomised) or ``"random"``.
    learn_snu : treat the inducing noise as a hyperparameter.
    """

    def __init__(
        self,
        nu: int,
        cov: str = DEFAULTS["sparse"]["cov"],
        degree: int = DEFAULTS["sparse"]["degree"],
        mean: str = DEFAULTS["sparse"]["mean"],
        lik: str = DEFAULTS["sparse"]["lik"],
        s: float = DEFAULTS["sparse"]["s"],
        init: str = DEFAULTS["sparse"]["init"],
        learn_snu: bool = DEFAULTS["sparse"]["learn_snu"],
        seed: int = DEFAULTS["sparse"]["seed"],
    ):
        self.nu, self.s, self.init, self.seed, self.learn_snu = nu, s, init, seed, learn_snu
        self.cov, self.mean, self.lik = (
            covariance(cov, degree),
            mean_function(mean),
            make_likelihood(lik),
        )
        self.inf = gaussian_posterior if isinstance(self.lik, gaussian) else laplace_posterior
        self.hyp: dict[str, Tensor] | None = None
        self.post: posterior_state | None = None
        self.nlZ: float | None = None

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> Self:
        """Construct from a flat JSON settings file, with optional keyword overrides."""
        return cls(**load_model_config(path, **overrides))

    # ----------------------------------------------------------------------- helpers
    def _initial_hyp(self, x: Tensor, y: Tensor, learn_xu: bool) -> dict[str, Tensor]:
        d = x.shape[1]
        ell = torch.log(x.std(0).clamp_min(DEFAULTS["initialization"]["lengthscale_floor"]))
        gauss = isinstance(self.lik, gaussian)
        sf = (
            torch.log(y.std().clamp_min(DEFAULTS["initialization"]["signal_std_floor"]))
            if gauss
            else torch.tensor(math.log(DEFAULTS["initialization"]["classification_signal_std"]))
        )
        hc = (
            torch.cat([ell, sf[None]])
            if self.cov.kind.endswith("ard")
            else torch.stack([ell.mean(), sf])
        )
        hm = torch.zeros(self.mean.num_hyp(d))
        if self.mean.kind in ("const", "linear+const") and gauss:
            hm[-1] = y.mean()
        hyp = make_hyperparameters(
            mean=hm,
            cov=hc,
            lik=torch.log(y.std() / DEFAULTS["initialization"]["sparse_noise_std_divisor"])[None]
            if gauss
            else torch.zeros(0),
        )
        xu = init_inducing(x, self.nu, self.init, self.seed)
        self._xu0 = xu
        if learn_xu:
            hyp["xu"] = xu
        if self.learn_snu:
            hyp["snu"] = torch.log(
                INFERENCE["sparse"]["inducing_noise_variance_ratio"] ** 0.5 * hyp["cov"][-1].exp()
            )[None]
        return hyp

    def _cov(self) -> sparse_covariance:
        return sparse_covariance(self.cov, self._xu0)

    # --------------------------------------------------------------------------- API
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        learn_xu: bool = DEFAULTS["sparse_fit"]["learn_xu"],
        swap: int = DEFAULTS["sparse_fit"]["swap"],
        max_iter: int = DEFAULTS["sparse_fit"]["max_iter"],
        hyp: dict[str, Tensor] | None = None,
    ) -> sparse_gp:
        """Fit through :mod:`gp_omni.train.sparse`."""
        from gp_omni.train.sparse import fit

        return fit(self, x, y, learn_xu, swap, max_iter, hyp)

    def predict(self, xs: Tensor) -> dict[str, Tensor]:
        """Predict through :mod:`gp_omni.infer.sparse`."""
        from gp_omni.infer.sparse import predict

        return predict(self, xs)

    @property
    def inducing_inputs(self) -> Tensor:
        if self.hyp is None:
            raise RuntimeError("call fit() first")
        return self.hyp["xu"].detach() if "xu" in self.hyp else self._xu0
