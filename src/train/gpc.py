"""Normalize binary labels and fit a GP classifier."""

import torch
from torch import Tensor

from gp_omni.settings import load_defaults
from gp_omni.train.gpr import fit as fit_gpr

DEFAULTS = load_defaults("models")


def as_labels(y: Tensor) -> Tensor:
    """Constraint labels as floats in {-1, +1}: bool, {0, 1} or {-1, +1} are accepted."""
    y = torch.as_tensor(y)
    if y.dtype == torch.bool:
        return torch.where(y, 1.0, -1.0).to(torch.float64)
    y = y.to(torch.float64).reshape(-1)
    values = set(y.unique().tolist())
    if values <= {-1.0, 1.0}:
        return y
    if values <= {0.0, 1.0}:
        return 2.0 * y - 1.0
    raise ValueError("constraint labels must be bool, {0, 1} or {-1, +1}")


def fit(
    model,
    x,
    y,
    output=None,
    fidelity=None,
    max_iter=DEFAULTS["gpc_fit"]["max_iter"],
    learn_xu=DEFAULTS["gpc_fit"]["learn_xu"],
):
    """Normalize labels and optimize approximate classification evidence."""
    return fit_gpr(model, x, as_labels(y), output, fidelity, max_iter, learn_xu)
