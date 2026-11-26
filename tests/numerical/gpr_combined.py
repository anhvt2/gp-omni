"""Pick any subset of the four features and run it end to end.

    python tests/numerical/gpr_combined.py                      # 1+2, then +3, then +4 (default)
    python tests/numerical/gpr_combined.py mo additive          # high-dimensional multi-objective
    python tests/numerical/gpr_combined.py mf sparse            # sparse multi-fidelity
    python tests/numerical/gpr_combined.py mo mf sparse additive    # or: all

Features: ``mo`` (correlated multi-objective), ``mf`` (multi-fidelity), ``sparse`` (big data,
sparse covariance), ``additive`` (additive high-dimensional). Any subset works, including one
feature or none; the data are generated to match the features you pick and the error is
measured at the highest fidelity of every output. The data layout is the same as in the
single-feature examples: one ``x``, one ``y``, plus an ``output`` and a ``fidelity`` index
per row.
"""

import sys

import torch

import gp_omni  # noqa: F401
from gp_omni import GPR

FEATURES = ("mo", "mf", "sparse", "additive")
W = [1.0, 0.8]  # how each objective loads on the shared structure


def truth(x, o, l):
    s = torch.sin(3 * x[:, 0]) + torch.cos(2 * x[:, 1]) + x[:, 2] ** 2 + 0.5 * x[:, 3]
    f = W[o] * s + 0.2 * o
    return f if l == 0 else 0.9 * f + 0.3 * torch.sin(5 * x[:, 0])


def run_features(features, max_iter: int = 80) -> float:
    """Fit and score the model with exactly the given features switched on.

    Returns the worst normalised RMSE over outputs, at the highest fidelity.
    """
    features = set(features)
    unknown = features - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}; choose from {FEATURES}")
    O = 2 if "mo" in features else 1
    L = 2 if "mf" in features else 1
    sizes = [200, 30] if L == 2 else [120]  # many cheap, few expensive

    g = torch.Generator().manual_seed(0)
    xs, ys, os_, fs = [], [], [], []
    for o in range(O):
        for l, n in enumerate(sizes):
            x = torch.rand(n, 4, generator=g)
            xs.append(x)
            ys.append(truth(x, o, l) + 0.02 * torch.randn(n, generator=g))
            os_.append(torch.full((n,), o))
            fs.append(torch.full((n,), l))
    x, y, o, f = torch.cat(xs), torch.cat(ys), torch.cat(os_), torch.cat(fs)
    xt = torch.rand(400, 4, generator=g)

    model = GPR(
        num_outputs=O,
        num_fidelities=L,
        nu=40 if "sparse" in features else None,
        additive=2 if "additive" in features else None,
    ).fit(x, y, o, f, max_iter=max_iter)
    pred = model.predict(xt)["fmu"].reshape(400, O)  # highest fidelity, every output
    errs = [
        ((pred[:, k] - truth(xt, k, L - 1)) ** 2).mean().sqrt() / truth(xt, k, L - 1).std()
        for k in range(O)
    ]
    return max(e.item() for e in errs)


def main(features=None) -> dict:
    """``features=None`` runs 1+2, then adds sparse, then additive; otherwise just the subset."""
    if features is None:
        steps = [
            ("mo+mf", {"mo", "mf"}),
            ("mo+mf+sparse", {"mo", "mf", "sparse"}),
            ("mo+mf+sparse+additive", set(FEATURES)),
        ]
    else:
        features = set(features)
        steps = [("+".join(f for f in FEATURES if f in features) or "none", features)]
    return {name: run_features(sel) for name, sel in steps}


if __name__ == "__main__":
    args = sys.argv[1:]
    print(main(set(FEATURES) if args == ["all"] else (args or None)))
