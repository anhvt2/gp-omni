"""Use case 2: multi-fidelity (cheap, plentiful low-fidelity data + a few expensive points).

Fidelity 0 is the cheap model, fidelity 1 the expensive one (the one you want to predict).
The model is autoregressive: f_1(x) = rho * f_0(x) + delta(x). Fidelity indices are integers,
ordered from cheapest (0) to most expensive.
"""

import torch

import gp_omni  # noqa: F401
from gp_omni import OmniGP


def main() -> dict:
    g = torch.Generator().manual_seed(0)
    hi = lambda x: (6 * x[:, 0] - 2) ** 2 * torch.sin(12 * x[:, 0] - 4)
    lo = lambda x: 0.5 * hi(x) + 10 * (x[:, 0] - 0.5) - 5
    xl, xh = torch.rand(30, 1, generator=g), torch.linspace(0, 1, 4).reshape(-1, 1)
    x = torch.cat([xl, xh])
    fidelity = torch.cat([torch.zeros(30), torch.ones(4)]).long()
    y = torch.cat([lo(xl), hi(xh)])

    model = OmniGP(num_fidelities=2).fit(x, y, fidelity=fidelity)  # <- the feature switch
    hi_only = OmniGP().fit(xh, hi(xh))

    xt = torch.linspace(0, 1, 200).reshape(-1, 1)
    rmse = lambda p: ((p - hi(xt)) ** 2).mean().sqrt().item()
    return {
        "mf_rmse": rmse(model.predict(xt)["fmu"]),  # predicts the highest fidelity by default
        "hi_only_rmse": rmse(hi_only.predict(xt)["fmu"]),
    }


if __name__ == "__main__":
    print(main())
