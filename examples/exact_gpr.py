"""Build, train and predict with an exact GP regression model."""

import torch

import gp_omni  # noqa: F401
from gp_omni.build.gpytorch import exact_gpr
from gp_omni.infer import predict
from gp_omni.train.gpytorch import fit_exact


def main() -> dict:
    torch.manual_seed(0)
    f = lambda x: torch.sin(6 * x[:, 0]) + x[:, 1] ** 2
    X, Xt = torch.rand(120, 2), torch.rand(200, 2)
    model = exact_gpr(X, f(X) + 0.05 * torch.randn(120))
    fit_exact(model, num_restarts=2)
    with torch.no_grad():
        rmse = ((predict(model, Xt)["fmu"] - f(Xt)) ** 2).mean().sqrt().item()
    return {"rmse": rmse}


if __name__ == "__main__":
    print(main())
