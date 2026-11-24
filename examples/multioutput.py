"""Correlated multi-output GP: ICM and LMC."""

import torch

import gp_omni  # noqa: F401
from gp_omni.build.gpytorch import multitask_gpr
from gp_omni.infer import predict
from gp_omni.train.gpytorch import fit_exact


def main() -> dict:
    torch.manual_seed(0)
    X, Xt = torch.rand(150, 1), torch.rand(200, 1)
    f = lambda x: torch.stack([torch.sin(6 * x[:, 0]), 0.5 * torch.sin(6 * x[:, 0]) + x[:, 0]], -1)
    out = {}
    for kind in ("icm", "lmc"):
        model = multitask_gpr(X, f(X) + 0.05 * torch.randn(150, 2), kind=kind)
        fit_exact(model)
        with torch.no_grad():
            out[f"{kind}_rmse"] = ((predict(model, Xt)["fmu"] - f(Xt)) ** 2).mean().sqrt().item()
    return out


if __name__ == "__main__":
    print(main())
