"""Predict with fitted GP models."""


def predict(model, x, *args, **kwargs):
    """Evaluate a fitted regression, classification, sparse, or co-kriging model."""
    return model.predict(x, *args, **kwargs)


__all__ = ["predict"]
