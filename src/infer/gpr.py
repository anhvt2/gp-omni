"""Predict task-specific moments and restore the original target scale."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.build.likelihood import gaussian
from gp_omni.build.multitask import (
    augment,
)
from gp_omni.infer.moments import predict_gp


def predict_one(model, xs: Tensor, o: int, f: int) -> dict[str, Tensor]:
    xa = augment(xs, torch.full((xs.shape[0],), float(o)), torch.full((xs.shape[0],), float(f)))
    xa = xa if model.multitask else xs
    out = predict_gp(
        model.hyp, model.post, model.mean, model._sparse_cov(model._xu0), model.lik, model._xa, xa
    )
    if not isinstance(model.lik, gaussian):  # classification: p(y = +1 | data)
        out["prob"] = torch.clamp((out["ymu"] + 1) / 2, 0.0, 1.0)
    if isinstance(model.lik, gaussian):  # undo the per-task standardisation
        k = o * model.L + f
        sd, mu = model._sd[k], model._mu[k]
        out = {
            "fmu": out["fmu"] * sd + mu,
            "ymu": out["ymu"] * sd + mu,
            "fs2": out["fs2"] * sd**2,
            "ys2": out["ys2"] * sd**2,
        }
    return out


def predict(
    model, xs: Tensor, output: int | None = None, fidelity: int | None = None
) -> dict[str, Tensor]:
    """Predictive ``fmu, fs2`` (latent) and ``ymu, ys2`` (with noise / class probability).

    ``fidelity`` defaults to the highest. With several outputs and ``output=None`` every
    entry has shape ``(m, num_outputs)``.
    """
    if model.hyp is None or model.post is None:
        raise RuntimeError("call fit() first")
    xs = xs.to(torch.float64)
    f = model.L - 1 if fidelity is None else fidelity
    if output is not None or model.O == 1:
        return predict_one(model, xs, output or 0, f)
    per = [predict_one(model, xs, k, f) for k in range(model.O)]
    return {key: torch.stack([p[key] for p in per], dim=1) for key in per[0]}
