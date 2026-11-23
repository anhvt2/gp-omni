"""Combining the features: they are independent, so any subset can be switched on together.

One problem generator, four feature switches. Runs 1+2 (multi-objective + multi-fidelity),
1+2+3 (+ sparse) and 1+2+3+4 (+ additive) on the same data and reports the error of each at
the highest fidelity. The data are laid out as in the single-feature examples: one ``x``, one
``y``, plus an ``output`` and a ``fidelity`` index per row.
"""

import torch

import gp_omni  # noqa: F401
from gp_omni import OmniGP

W = [1.0, 0.8]  # how each objective loads on the shared structure


def truth(x, o, l):
    s = torch.sin(3 * x[:, 0]) + torch.cos(2 * x[:, 1]) + x[:, 2] ** 2 + 0.5 * x[:, 3]
    f = W[o] * s + 0.2 * o
    return f if l == 0 else 0.9 * f + 0.3 * torch.sin(5 * x[:, 0])


def main() -> dict:
    g = torch.Generator().manual_seed(0)
    xs, ys, os_, fs = [], [], [], []
    for o in range(2):
        for l, n in enumerate([300, 40]):  # many cheap, few expensive
            x = torch.rand(n, 4, generator=g)
            xs.append(x)
            ys.append(truth(x, o, l) + 0.02 * torch.randn(n, generator=g))
            os_.append(torch.full((n,), o))
            fs.append(torch.full((n,), l))
    x, y, o, f = torch.cat(xs), torch.cat(ys), torch.cat(os_), torch.cat(fs)
    xt = torch.rand(400, 4, generator=g)

    configs = {
        "mo+mf": {},
        "mo+mf+sparse": {"nu": 40},
        "mo+mf+sparse+additive": {"nu": 40, "additive": 2},
    }
    out = {}
    for name, kw in configs.items():
        model = OmniGP(num_outputs=2, num_fidelities=2, **kw).fit(x, y, o, f, max_iter=80)
        pred = model.predict(xt)["fmu"]  # (400, 2): highest fidelity, every output
        errs = [
            ((pred[:, k] - truth(xt, k, 1)) ** 2).mean().sqrt() / truth(xt, k, 1).std()
            for k in range(2)
        ]
        out[name] = max(e.item() for e in errs)
    return out


if __name__ == "__main__":
    print(main())
