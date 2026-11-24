"""Prediction for fitted sparse Gaussian processes."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.infer.moments import predict_gp


def predict(model, xs: Tensor) -> dict[str, Tensor]:
    """gpml's ``[ymu, ys2, fmu, fs2]`` (dictionary)."""
    if model.hyp is None or model.post is None:
        raise RuntimeError("call fit() first")
    xs = xs.to(torch.float64)
    return predict_gp(model.hyp, model.post, model.mean, model._cov(), model.lik, model._x, xs)
