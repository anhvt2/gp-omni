"""Fit GP hyperparameters by likelihood maximization or variational optimization."""


def fit(model, *args, **kwargs):
    """Train a prepared GPR, GPC, sparse GP, or co-kriging model and return it.

    GPyTorch models use ``fit_exact`` or ``fit_variational`` from ``train.gpytorch``.
    """
    return model.fit(*args, **kwargs)


__all__ = ["fit"]
