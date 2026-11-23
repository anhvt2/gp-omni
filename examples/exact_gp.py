"""Exact GP regression with a gpml-style kernel spec."""

import torch

import gp_omni  # noqa: F401
from gp_omni.builders import exact_gp
from gp_omni.compat import gpml_kernel
from gp_omni.fit import fit_exact


def main() -> dict:
    torch.manual_seed(0)
    f = lambda x: torch.sin(6 * x[:, 0]) + x[:, 1] ** 2
    X, Xt = torch.rand(120, 2), torch.rand(200, 2)
    model = exact_gp(X, f(X) + 0.05 * torch.randn(120), gpml_kernel(("covMaternard", 5), 2))
    fit_exact(model, num_restarts=2)
    with torch.no_grad():
        rmse = ((model(Xt).mean - f(Xt)) ** 2).mean().sqrt().item()
    return {"rmse": rmse}


if __name__ == "__main__":
    print(main())
