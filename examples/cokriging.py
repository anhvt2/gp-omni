"""AR1 co-kriging (port of the owner's MATLAB @CoKriging) on the Forrester 1-D benchmark."""

import torch

import gp_omni  # noqa: F401
from gp_omni import ar1_cokriging


def main() -> dict:
    hi = lambda x: (6 * x - 2) ** 2 * torch.sin(12 * x - 4)
    lo = lambda x: 0.5 * hi(x) + 10 * (x - 0.5) - 5
    xl, xh = torch.linspace(0, 1, 20).reshape(-1, 1), torch.linspace(0, 1, 4).reshape(-1, 1)
    X, Y = [xl, xh], [lo(xl), hi(xh)]
    model = ar1_cokriging(X, Y, correlation="matern32")
    # rho = 0 switches the low-fidelity data off: kriging on the 4 high-fidelity points only
    hi_only = ar1_cokriging(
        X, Y, correlation="matern32", hyperparameters={**model.hyperparameters, "rho": 0.0}
    )
    xt = torch.linspace(0, 1, 200).reshape(-1, 1)
    rmse = lambda m: ((m.predict(xt)[0] - hi(xt[:, 0])) ** 2).mean().sqrt().item()
    return {"rmse": rmse(model), "hi_only_rmse": rmse(hi_only), "rho": model.rho}


if __name__ == "__main__":
    print(main())
