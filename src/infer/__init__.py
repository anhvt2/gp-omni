"""Evaluate trained GPs and compute their predictive distributions."""


def predict(model, x, *args, **kwargs):
    """Predict with a fitted GPR, GPC, sparse GP, co-kriging, or GPyTorch model."""
    if callable(getattr(model, "predict", None)):
        return model.predict(x, *args, **kwargs)
    from gp_omni.infer.gpytorch import predict as predict_gpytorch

    return predict_gpytorch(model, x, *args, **kwargs)


__all__ = ["predict"]
