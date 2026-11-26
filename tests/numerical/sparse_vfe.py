"""Sparse variational GP (VFE vs FITC).

VFE (``s=0``) vs FITC (``s=1``) on 3000 noisy points with 30 inducing inputs, plus a probit
classifier on the same sparse approximation. The true noise standard deviation is 0.1.
"""

import torch

import gp_omni  # noqa: F401
from gp_omni.build.sparse import sparse_gp


def main() -> dict:
    g = torch.Generator().manual_seed(0)
    f = lambda x: torch.sin(6 * x[:, 0]) + x[:, 1] ** 2
    x, xt = torch.rand(3000, 2, generator=g), torch.rand(500, 2, generator=g)
    y = f(x) + 0.1 * torch.randn(3000, generator=g)
    out = {}
    for name, s in (("vfe", 0.0), ("fitc", 1.0)):
        model = sparse_gp(30, s=s).fit(x, y, swap=0)
        pred = model.predict(xt)
        out[f"{name}_rmse"] = ((pred["fmu"] - f(xt)) ** 2).mean().sqrt().item()
        out[f"{name}_noise_sd"] = model.hyp["lik"].exp().item()
    rule = lambda z: torch.where(z[:, 0] + z[:, 1] > 1, 1.0, -1.0)
    clf = sparse_gp(20, lik="probit").fit(x[:500], rule(x[:500]), max_iter=80)
    out["probit_accuracy"] = (torch.sign(clf.predict(xt)["ymu"]) == rule(xt)).double().mean().item()
    return out


if __name__ == "__main__":
    print(main())
