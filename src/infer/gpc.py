"""Probability and feasibility decisions for a trained GP classifier."""

from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("models")


def predict_proba(model, xs, output=None, fidelity=None):
    """Return the marginal probability of a positive/feasible class label."""
    return model.predict(xs, output, fidelity)["prob"]


def predict_feasible(
    model, xs, output=None, fidelity=None, threshold=DEFAULTS["decision"]["threshold"]
):
    """Threshold the marginal feasibility probability."""
    return predict_proba(model, xs, output, fidelity) >= threshold
