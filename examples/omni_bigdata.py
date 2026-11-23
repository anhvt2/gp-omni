"""Use case 3: big data with a sparse covariance approximation (gpml ``apxSparse``).

20,000 points are fit through 40 inducing inputs: O(n nu^2) instead of O(n^3). ``s`` picks the
method: 0 = variational free energy (default), 1 = FITC, in between = sparse power EP.
"""

import torch

import gp_omni  # noqa: F401
from gp_omni import OmniGP


def main() -> dict:
    g = torch.Generator().manual_seed(0)
    f = lambda x: torch.sin(6 * x[:, 0]) + x[:, 1] ** 2
    x = torch.rand(20_000, 2, generator=g)
    y = f(x) + 0.1 * torch.randn(20_000, generator=g)
    out = {}
    for name, s in (("vfe", 0.0), ("fitc", 1.0)):
        model = OmniGP(nu=40, s=s).fit(x, y, max_iter=30)  # <- the feature switch
        xt = torch.rand(500, 2, generator=g)
        out[f"{name}_rmse"] = ((model.predict(xt)["fmu"] - f(xt)) ** 2).mean().sqrt().item()
        out[f"{name}_noise_sd"] = model.hyp["lik"].exp().item() * model._sd[0].item()  # true: 0.1
    return out


if __name__ == "__main__":
    print(main())
