"""Multi-fidelity with BoTorch's SingleTaskMultiFidelityGP (fidelity is an input column).

Compared against a single-fidelity GP trained on the four high-fidelity points only.
"""

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, SingleTaskMultiFidelityGP
from gpytorch.mlls import ExactMarginalLogLikelihood

import gp_omni  # noqa: F401


def main() -> dict:
    hi = lambda x: (6 * x - 2) ** 2 * torch.sin(12 * x - 4)
    lo = lambda x: 0.5 * hi(x) + 10 * (x - 0.5) - 5
    xl, xh = torch.linspace(0, 1, 20).reshape(-1, 1), torch.linspace(0, 1, 4).reshape(-1, 1)
    X = torch.cat([torch.cat([xl, torch.zeros(20, 1)], 1), torch.cat([xh, torch.ones(4, 1)], 1)])
    mf = SingleTaskMultiFidelityGP(X, torch.cat([lo(xl), hi(xh)]), data_fidelities=[1])
    fit_gpytorch_mll(ExactMarginalLogLikelihood(mf.likelihood, mf))
    sf = SingleTaskGP(xh, hi(xh))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(sf.likelihood, sf))
    xt = torch.linspace(0, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        m_mf = mf.posterior(torch.cat([xt, torch.ones(100, 1)], 1)).mean.squeeze(-1)
        m_sf = sf.posterior(xt).mean.squeeze(-1)
    rmse = lambda m: ((m - hi(xt[:, 0])) ** 2).mean().sqrt().item()
    return {"rmse": rmse(m_mf), "hi_only_rmse": rmse(m_sf)}


if __name__ == "__main__":
    print(main())
