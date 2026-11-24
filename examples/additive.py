"""Additive GP  vs plain ARD on a purely additive 20-D function."""

import torch

import gp_omni  # noqa: F401
from gp_omni.build.gpytorch import additive_gpr, exact_gpr
from gp_omni.infer import predict
from gp_omni.train.gpytorch import fit_exact


def main() -> dict:
    torch.manual_seed(0)
    d = 20
    w = torch.randn(d)
    f = lambda x: sum(w[i] * torch.sin(3 * x[:, i] + i) for i in range(d))
    X, Xt = torch.rand(120, d), torch.rand(300, d)
    y = f(X) + 0.02 * torch.randn(120)
    out = {}
    for name, m in (("additive", additive_gpr(X, y, max_degree=2)), ("plain_ard", exact_gpr(X, y))):
        fit_exact(m)
        with torch.no_grad():
            err = ((predict(m, Xt)["fmu"] - f(Xt)) ** 2).mean().sqrt() / f(Xt).std()
        out[f"{name}_nrmse"] = err.item()
    return out


if __name__ == "__main__":
    print(main())
