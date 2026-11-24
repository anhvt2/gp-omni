"""Predictive moments for trained GPyTorch regression and classification models."""

import torch


def predict(model, x, *, likelihood=None):
    """Return latent and observed moments without fitting model parameters.

    Supply the separately trained likelihood for a variational model. Exact models
    carry their likelihood. Bernoulli observed means are probabilities in [0, 1].
    """
    model.eval()
    observation = likelihood if likelihood is not None else getattr(model, "likelihood", None)
    if observation is not None:
        observation.eval()
    with torch.no_grad():
        latent = model(x)
        observed = latent if observation is None else observation(latent)
        return {
            "fmu": latent.mean,
            "fs2": latent.variance,
            "ymu": observed.mean,
            "ys2": observed.variance,
        }
