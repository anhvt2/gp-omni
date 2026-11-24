"""End-to-end sparse variational GP: initialise, (optionally) swap, train, predict.

The default ``s = 0`` is the variational free energy bound (Titsias 2009); ``s = 1`` gives
FITC and ``0 < s < 1`` sparse power EP, exactly as ``opt.s`` in gpml's ``apx``.
Gaussian likelihoods use ``infGaussLik``; ``likErf`` / ``likLogistic`` (labels +/-1) use
``infLaplace`` on the same sparse covariance approximation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Self

import torch
from torch import Tensor

from gp_omni.settings import load_defaults, load_model_config
from gp_omni.sparse.apx import SparseCov
from gp_omni.sparse.cov import Cov
from gp_omni.sparse.gp import gp_nlz, gp_predict, make_hyp
from gp_omni.sparse.inf import Posterior, inf_gauss_lik, inf_laplace
from gp_omni.sparse.init import init_inducing
from gp_omni.sparse.lik import LikGauss, make_lik
from gp_omni.sparse.mean import Mean
from gp_omni.sparse.train import train
from gp_omni.sparse.xu_opt import vfe_xu_opt

DEFAULTS = load_defaults("models")
INFERENCE = load_defaults("inference")


class SparseGP:
    """Sparse GP regression / classification following gpml's ``apxSparse``.

    Parameters
    ----------
    nu : number of inducing inputs.
    cov, degree : ``"SEard"``, ``"SEiso"``, ``"Maternard"``, ``"Materniso"`` (+ Matern degree).
    mean : ``"zero"``, ``"const"``, ``"linear"`` or ``"linear+const"``.
    lik : ``"likGauss"``, ``"likErf"`` or ``"likLogistic"``.
    s : sparse power in ``[0, 1]``: ``0`` VFE (default), ``1`` FITC.
    init : ``"kmeans"`` (Improved-Nystrom landmarks, randomised) or ``"random"``.
    learn_snu : treat the inducing noise as a hyperparameter (gpml ``hyp.snu``).
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
        self.cov, self.mean, self.lik = Cov(cov, degree), Mean(mean), make_lik(lik)
        self.inf = inf_gauss_lik if isinstance(self.lik, LikGauss) else inf_laplace
        self.hyp: dict[str, Tensor] | None = None
        self.post: Posterior | None = None
        self.nlZ: float | None = None

    @classmethod
    def from_json(cls, path: str | Path, **overrides) -> Self:
        """Construct from a flat JSON settings file, with optional keyword overrides."""
        return cls(**load_model_config(path, **overrides))

    # ----------------------------------------------------------------------- helpers
    def _initial_hyp(self, x: Tensor, y: Tensor, learn_xu: bool) -> dict[str, Tensor]:
        d = x.shape[1]
        ell = torch.log(x.std(0).clamp_min(DEFAULTS["initialization"]["lengthscale_floor"]))
        gauss = isinstance(self.lik, LikGauss)
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
        hyp = make_hyp(
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

    def _cov(self) -> SparseCov:
        return SparseCov(self.cov, self._xu0)

    # --------------------------------------------------------------------------- API
    def fit(
        self,
        x: Tensor,
        y: Tensor,
        learn_xu: bool = DEFAULTS["sparse_fit"]["learn_xu"],
        swap: int = DEFAULTS["sparse_fit"]["swap"],
        max_iter: int = DEFAULTS["sparse_fit"]["max_iter"],
        hyp: dict[str, Tensor] | None = None,
    ) -> SparseGP:
        """Train hyperparameters (and inducing inputs if ``learn_xu``) on ``(x, y)``.

        ``swap > 0`` first runs gpml's greedy ``vfe_xu_opt`` swaps (``s = 0``, Gaussian
        likelihood only), with the training inputs as candidates.
        """
        x, y = x.to(torch.float64), y.to(torch.float64).reshape(-1)
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        if not isinstance(self.lik, LikGauss) and not set(y.unique().tolist()) <= {-1.0, 1.0}:
            raise ValueError("classification labels must be in {-1, +1}")
        h = hyp or self._initial_hyp(x, y, learn_xu)
        if hyp is not None:
            self._xu0 = h.get("xu", init_inducing(x, self.nu, self.init, self.seed))
        opt = {"s": self.s}
        if swap > 0:
            if not (isinstance(self.lik, LikGauss) and self.s == 0.0):
                raise ValueError("vfe_xu_opt swaps need s=0 and a Gaussian likelihood")
            h["xu"] = self._xu0.clone()
            h, _ = vfe_xu_opt(h, self.mean, self.cov, x, y, x, swap)
            self._xu0 = h["xu"].detach()
            if not learn_xu:
                h.pop("xu")
        h, nlZ = train(h, self.inf, self.mean, self._cov(), self.lik, x, y, opt, max_iter)
        self.hyp, self.nlZ, self._x, self._y, self._opt = h, nlZ, x, y, opt
        _, _, self.post = gp_nlz(
            h, self.inf, self.mean, self._cov(), self.lik, x, y, opt, grad=False
        )
        return self

    def predict(self, xs: Tensor) -> dict[str, Tensor]:
        """gpml's ``[ymu, ys2, fmu, fs2]`` (dictionary)."""
        if self.hyp is None or self.post is None:
            raise RuntimeError("call fit() first")
        xs = xs.to(torch.float64)
        return gp_predict(self.hyp, self.post, self.mean, self._cov(), self.lik, self._x, xs)

    @property
    def inducing_inputs(self) -> Tensor:
        if self.hyp is None:
            raise RuntimeError("call fit() first")
        return self.hyp["xu"].detach() if "xu" in self.hyp else self._xu0
