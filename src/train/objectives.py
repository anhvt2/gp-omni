"""Differentiable negative log evidence for exact, sparse and Laplace GPs."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from gp_omni.build.covariance import covariance
from gp_omni.build.hyperparameters import check_hyperparameters
from gp_omni.build.likelihood import likelihood
from gp_omni.build.mean import mean_function
from gp_omni.build.sparse_covariance import sparse_covariance
from gp_omni.infer.posterior import posterior_state

inference_method = Callable[..., tuple[posterior_state, Tensor]]


def negative_log_evidence(
    hyp: dict,
    inf: inference_method,
    mean: mean_function,
    cov: covariance | sparse_covariance,
    lik: likelihood,
    x: Tensor,
    y: Tensor,
    opt: dict | None = None,
    grad: bool = True,
) -> tuple[Tensor, dict[str, Tensor] | None, posterior_state]:
    """``[nlZ, dnlZ, post] = gp(hyp, inf, mean, cov, lik, x, y)``.

    ``dnlZ`` has the same keys as ``hyp`` (all tensors in ``hyp`` that require grad are
    differentiated; others are treated as constants)."""
    check_hyperparameters(hyp, mean, cov, lik, x.shape[1])
    post, nlZ = inf(hyp, mean, cov, lik, x, y, opt)
    dnlZ = None
    if grad:
        keys = [k for k, v in hyp.items() if v.requires_grad]
        gr = torch.autograd.grad(nlZ, [hyp[k] for k in keys], allow_unused=True)
        dnlZ = {k: (torch.zeros_like(hyp[k]) if g is None else g) for k, g in zip(keys, gr)}
    return nlZ, dnlZ, post
