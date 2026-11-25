"""Predict task-specific moments and restore the original target scale."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.build.likelihood import gaussian
from gp_omni.build.multitask import (
    augment,
)
from gp_omni.infer.moments import predict_gp


def predict_one(model, xs: Tensor, o: int, f: int, full_cov: bool = False) -> dict[str, Tensor]:
    xa = augment(xs, torch.full((xs.shape[0],), float(o)), torch.full((xs.shape[0],), float(f)))
    xa = xa if model.multitask else xs
    out = predict_gp(
        model.hyp,
        model.post,
        model.mean,
        model._sparse_cov(model._xu0),
        model.lik,
        model._xa,
        xa,
        full_cov=full_cov,
    )
    if not isinstance(model.lik, gaussian):  # classification: p(y = +1 | data)
        out["prob"] = torch.clamp((out["ymu"] + 1) / 2, 0.0, 1.0)
    if isinstance(model.lik, gaussian):  # undo the per-task standardisation
        k = o * model.L + f
        sd, mu = model._sd[k], model._mu[k]
        scaled = {
            "fmu": out["fmu"] * sd + mu,
            "ymu": out["ymu"] * sd + mu,
            "fs2": out["fs2"] * sd**2,
            "ys2": out["ys2"] * sd**2,
        }
        if "fcov" in out:
            scaled["fcov"] = out["fcov"] * sd**2
        out = scaled
    return out


def predict(
    model,
    xs: Tensor,
    output: int | None = None,
    fidelity: int | None = None,
    full_cov: bool = False,
) -> dict[str, Tensor]:
    """Predictive ``fmu, fs2`` (latent) and ``ymu, ys2`` (with noise / class probability).

    ``fidelity`` defaults to the highest. With several outputs and ``output=None`` every
    entry has shape ``(m, num_outputs)``.

    ``full_cov=True`` adds ``fcov``, the latent joint covariance over ``xs`` (one output only).
    """
    if model.hyp is None or model.post is None:
        raise RuntimeError("call fit() first")
    xs = xs.to(torch.float64)
    f = model.L - 1 if fidelity is None else fidelity
    if full_cov and output is None and model.O > 1:
        raise NotImplementedError("full_cov needs a single output; pass `output=`")
    if output is not None or model.O == 1:
        return predict_one(model, xs, output or 0, f, full_cov)
    per = [predict_one(model, xs, k, f) for k in range(model.O)]
    return {key: torch.stack([p[key] for p in per], dim=1) for key in per[0]}
