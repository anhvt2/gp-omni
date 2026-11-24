"""Big-data GPs: SGPR (inducing points), svgp (minibatch ELBO) and KISS-GP (grid)."""

import gpytorch
import torch

import gp_omni  # noqa: F401
from gp_omni.build.gpytorch import kiss_gpr, sgpr, svgp
from gp_omni.train.gpytorch import fit_exact, fit_variational


def main() -> dict:
    torch.manual_seed(0)
    f = lambda x: torch.sin(8 * x[:, 0])
    X, Xt = torch.rand(1500, 1), torch.rand(300, 1)
    y = f(X) + 0.05 * torch.randn(1500)
    out = {}
    m = sgpr(X, y, num_inducing=25)
    fit_exact(m)
    with torch.no_grad():
        out["sgpr_rmse"] = ((m.likelihood(m(Xt)).mean - f(Xt)) ** 2).mean().sqrt().item()
    m = kiss_gpr(X, y, grid_size=100)
    fit_exact(m)
    with torch.no_grad():
        out["kiss_rmse"] = ((m.likelihood(m(Xt)).mean - f(Xt)) ** 2).mean().sqrt().item()
    # Cover the domain before learning inducing locations; random sites can cluster.
    inducing = torch.linspace(0, 1, 25).reshape(-1, 1)
    sv, lik = svgp(inducing), gpytorch.likelihoods.GaussianLikelihood()
    fit_variational(sv, lik, X, y, epochs=500, lr=0.03)
    with torch.no_grad():
        out["svgp_rmse"] = ((sv(Xt).mean - f(Xt)) ** 2).mean().sqrt().item()
    return out


if __name__ == "__main__":
    print(main())
