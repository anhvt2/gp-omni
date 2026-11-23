"""A constraint classifier that mirrors the objective model, feature for feature.

    python examples/omni_constraints.py                       # no feature switches
    python examples/omni_constraints.py mo additive           # correlated + additive
    python examples/omni_constraints.py mo mf sparse additive # everything (or: all)

Features: ``mo`` (several correlated outputs), ``mf`` (several fidelities), ``sparse``
(inducing points), ``additive`` (additive covariance). The same switches configure both
twins: ``OmniGP`` regresses the objective, ``OmniGPC`` classifies the constraint, so the
two models share structure (outputs, fidelities, kernel, sparsity) and each feature can be
changed independently. Labels are +1 = feasible, -1 = infeasible.

The score is the objective's normalised RMSE and the constraint's accuracy, both at the
highest fidelity and worst over outputs.
"""

import sys

import torch

import gp_omni  # noqa: F401
from gp_omni import OmniGP, OmniGPC

FEATURES = ("mo", "mf", "sparse", "additive")
W = [1.0, 0.8]  # how each output loads on the shared structure
THRESHOLD = 0.3  # feasible when the constraint value exceeds this


def shared(x, o):
    s = torch.sin(3 * x[:, 0]) + torch.cos(2 * x[:, 1]) + x[:, 2] ** 2 + 0.5 * x[:, 3]
    return W[o] * s + 0.2 * o


def objective(x, o, l):
    f = shared(x, o)
    return f if l == 0 else 0.9 * f + 0.3 * torch.sin(5 * x[:, 0])


def constraint_value(x, o, l):
    g = 0.6 * shared(x, o) + 0.5 * torch.sin(4 * x[:, 1])
    return g if l == 0 else 0.9 * g + 0.2 * torch.cos(5 * x[:, 0])


def run_features(features, max_iter: int = 80) -> dict:
    """Fit the objective regressor and the constraint classifier with the same features."""
    features = set(features)
    unknown = features - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}; choose from {FEATURES}")
    O, L = (2 if "mo" in features else 1), (2 if "mf" in features else 1)
    switches = {
        "num_outputs": O,
        "num_fidelities": L,
        "nu": 40 if "sparse" in features else None,
        "additive": 2 if "additive" in features else None,
    }
    sizes = [200, 40] if L == 2 else [150]  # many cheap, few expensive

    g = torch.Generator().manual_seed(0)
    xs, ys, labels, os_, fs = [], [], [], [], []
    for o in range(O):
        for l, n in enumerate(sizes):
            x = torch.rand(n, 4, generator=g)
            xs.append(x)
            ys.append(objective(x, o, l) + 0.02 * torch.randn(n, generator=g))
            labels.append(torch.where(constraint_value(x, o, l) > THRESHOLD, 1.0, -1.0))
            os_.append(torch.full((n,), o))
            fs.append(torch.full((n,), l))
    x, y, c, o, f = (torch.cat(t) for t in (xs, ys, labels, os_, fs))
    xt = torch.rand(500, 4, generator=g)

    gpr = OmniGP(**switches).fit(x, y, o, f, max_iter=max_iter)  # objective
    gpc = OmniGPC(**switches).fit(x, c, o, f, max_iter=max_iter)  # constraint: same switches
    mu = gpr.predict(xt)["fmu"].reshape(500, O)
    p = gpc.predict_proba(xt).reshape(500, O)  # P(feasible) at the highest fidelity
    nrmse = [
        (((mu[:, k] - objective(xt, k, L - 1)) ** 2).mean().sqrt() / objective(xt, k, L - 1).std())
        for k in range(O)
    ]
    acc = [
        ((p[:, k] >= 0.5) == (constraint_value(xt, k, L - 1) > THRESHOLD)).double().mean()
        for k in range(O)
    ]
    return {
        "objective_nrmse": max(v.item() for v in nrmse),
        "constraint_accuracy": min(v.item() for v in acc),
    }


def main(features=None) -> dict:
    """``features=None`` runs a few representative subsets; otherwise just the one given."""
    if features is None:
        picks = [set(), {"mo", "additive"}, set(FEATURES)]
    else:
        picks = [set(features)]
    return {"+".join(f for f in FEATURES if f in sel) or "none": run_features(sel) for sel in picks}


if __name__ == "__main__":
    args = sys.argv[1:]
    print(main(set(FEATURES) if args == ["all"] else (args or None)))
