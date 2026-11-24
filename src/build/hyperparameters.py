"""Construct and validate Gaussian-process hyperparameter tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.build.covariance import covariance
from gp_omni.build.likelihood import likelihood
from gp_omni.build.mean import mean_function
from gp_omni.build.sparse_covariance import sparse_covariance


def make_hyperparameters(**kw) -> dict[str, Tensor]:
    """Build ``hyp`` (gpml's struct): float64 tensors for ``mean``, ``cov``, ``lik`` and
    optional ``xu`` (nu x D) / ``snu``."""
    return {k: _t(k, v) for k, v in kw.items()}


def _t(key: str, v) -> Tensor:
    t = torch.as_tensor(v, dtype=torch.float64)
    return t if key == "xu" else t.reshape(-1)


def check_hyperparameters(
    hyp: dict, mean: mean_function, cov: covariance | sparse_covariance, lik: likelihood, dim: int
) -> None:
    """gpml's hyperparameter-count checks."""
    base = cov.cov if isinstance(cov, sparse_covariance) else cov
    for key, need in (
        ("mean", mean.num_hyp(dim)),
        ("cov", base.num_hyp(dim)),
        ("lik", lik.num_hyp()),
    ):
        if hyp.get(key, torch.empty(0)).numel() != need:
            raise ValueError(f"Number of {key} hyperparameters disagree with the {key} function")
