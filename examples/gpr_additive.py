"""Use case 4: additive GP for high-dimensional regression (additive kernel).

A 20-D function that is a sum of 1-D effects, 120 samples. ``additive=R`` models interactions
up to order R (``additive=1`` is purely additive); a plain ARD kernel cannot exploit the
structure and fails with so few points.
"""

import torch

import gp_omni  # noqa: F401
from gp_omni import GPR


def main() -> dict:
    g = torch.Generator().manual_seed(0)
    d = 20
    w = torch.randn(d, generator=g)
    f = lambda x: sum(w[i] * torch.sin(3 * x[:, i] + i) for i in range(d))
    x, xt = torch.rand(120, d, generator=g), torch.rand(300, d, generator=g)
    y = f(x) + 0.02 * torch.randn(120, generator=g)
    out = {}
    for name, add in (("additive", 2), ("plain_ard", None)):
        model = GPR(additive=add).fit(x, y)  # <- the feature switch
        out[f"{name}_nrmse"] = (
            ((model.predict(xt)["fmu"] - f(xt)) ** 2).mean().sqrt() / f(xt).std()
        ).item()
    return out


if __name__ == "__main__":
    print(main())
