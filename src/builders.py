"""Small builders for the GP flavours GPyTorch/BoTorch ship, so each is a one-liner.

Everything returned is a plain GPyTorch / BoTorch object; gp-omni adds no wrapper types here.
"""

from __future__ import annotations

import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.kernels import (
    GridInterpolationKernel,
    InducingPointKernel,
    Kernel,
    LCMKernel,
    MaternKernel,
    MultitaskKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import Tensor

from gp_omni.kernels import AdditiveKernel
from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("builders")


class ExactGPModel(ExactGP, GPyTorchModel):
    """Exact GP with a constant mean and any kernel (also used for SGPR and KISS-GP).

    A BoTorch model, so it works with BoTorch acquisition functions.
    """

    _num_outputs = 1

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood, covar_module: Kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = covar_module

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def exact_gp(train_x: Tensor, train_y: Tensor, kernel: Kernel | None = None) -> ExactGPModel:
    """Exact GP; default kernel is ARD Matern-5/2 (``covMaternard`` with degree 5)."""
    kernel = kernel or ScaleKernel(
        MaternKernel(nu=DEFAULTS["exact"]["matern_nu"], ard_num_dims=train_x.shape[-1])
    )
    return ExactGPModel(train_x, train_y, GaussianLikelihood(), kernel)


def additive_gp(
    train_x: Tensor, train_y: Tensor, max_degree: int = DEFAULTS["additive"]["max_degree"]
) -> ExactGPModel:
    """Additive GP of interaction orders 1..``max_degree`` (gpml ``covADD`` with ``1:R``).

    Uses :class:`gp_omni.kernels.AdditiveKernel` (one learned output scale per order).
    """
    kern = AdditiveKernel(train_x.shape[-1], max_degree)
    return ExactGPModel(train_x, train_y, GaussianLikelihood(), kern)


def sgpr(
    train_x: Tensor, train_y: Tensor, num_inducing: int, seed: int = DEFAULTS["sparse"]["seed"]
) -> ExactGPModel:
    """Sparse GP regression, collapsed (Titsias VFE-style) via ``InducingPointKernel``."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(train_x.shape[0], generator=gen)[:num_inducing]
    lik = GaussianLikelihood()
    base = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[-1]))
    kern = InducingPointKernel(base, inducing_points=train_x[idx].clone(), likelihood=lik)
    return ExactGPModel(train_x, train_y, lik, kern)


def kiss_gp(
    train_x: Tensor, train_y: Tensor, grid_size: int = DEFAULTS["grid"]["grid_size"]
) -> ExactGPModel:
    """Structured kernel interpolation (grid) GP; intended for low input dimension (<= ~3)."""
    d = train_x.shape[-1]
    kern = GridInterpolationKernel(ScaleKernel(RBFKernel()), grid_size=grid_size, num_dims=d)
    return ExactGPModel(train_x, train_y, GaussianLikelihood(), kern)


class MultitaskExactGP(ExactGP):
    """Correlated multi-output exact GP: ``"icm"`` (intrinsic) or ``"lmc"`` (linear mixture)."""

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        kind: str = DEFAULTS["multitask"]["kind"],
        rank: int = DEFAULTS["multitask"]["rank"],
    ):
        num_tasks = train_y.shape[-1]
        lik = MultitaskGaussianLikelihood(num_tasks=num_tasks)
        super().__init__(train_x, train_y, lik)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        if kind == "icm":
            self.covar_module = MultitaskKernel(RBFKernel(), num_tasks=num_tasks, rank=rank)
        elif kind == "lmc":
            self.covar_module = LCMKernel(
                [RBFKernel(), MaternKernel(nu=DEFAULTS["lmc"]["matern_nu"])],
                num_tasks=num_tasks,
                rank=rank,
            )
        else:
            raise ValueError("kind must be 'icm' or 'lmc'")

    def forward(self, x: Tensor):
        return MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x))


class SVGP(ApproximateGP):
    """Sparse variational GP; with a Bernoulli likelihood it is the classification model."""

    def __init__(self, inducing_points: Tensor, kernel: Kernel | None = None):
        m, d = inducing_points.shape
        dist = CholeskyVariationalDistribution(m)
        strat = VariationalStrategy(
            self,
            inducing_points,
            dist,
            learn_inducing_locations=DEFAULTS["variational"]["learn_inducing_locations"],
        )
        super().__init__(strat)
        self.mean_module = ConstantMean()
        self.covar_module = kernel or ScaleKernel(RBFKernel(ard_num_dims=d))

    def forward(self, x: Tensor) -> MultivariateNormal:
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))
