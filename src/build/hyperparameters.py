"""Construct and validate Gaussian-process hyperparameter tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.build.covariance import Cov
from gp_omni.build.likelihood import Lik
from gp_omni.build.mean import Mean
from gp_omni.build.sparse_covariance import SparseCov


def make_hyperparameters(**kw) -> dict[str, Tensor]:
    """Build ``hyp`` (gpml's struct): float64 tensors for ``mean``, ``cov``, ``lik`` and
    optional ``xu`` (nu x D) / ``snu``."""
    return {k: _t(k, v) for k, v in kw.items()}


def _t(key: str, v) -> Tensor:
    t = torch.as_tensor(v, dtype=torch.float64)
    return t if key == "xu" else t.reshape(-1)


def check_hyperparameters(hyp: dict, mean: Mean, cov: Cov | SparseCov, lik: Lik, dim: int) -> None:
    """gpml's hyperparameter-count checks."""
    base = cov.cov if isinstance(cov, SparseCov) else cov
    for key, need in (
        ("mean", mean.num_hyp(dim)),
        ("cov", base.num_hyp(dim)),
        ("lik", lik.num_hyp()),
    ):
        if hyp.get(key, torch.empty(0)).numel() != need:
            raise ValueError(f"Number of {key} hyperparameters disagree with the {key} function")
