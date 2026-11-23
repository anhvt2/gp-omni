"""Binary GP classification (probit likelihood, variational inference).

gpml uses EP/Laplace for ``likErf``; GPyTorch has no EP, so this is the SVGP counterpart.
"""

import torch
from gpytorch.likelihoods import BernoulliLikelihood

import gp_omni  # noqa: F401
from gp_omni.builders import SVGP
from gp_omni.fit import fit_variational


def main() -> dict:
    torch.manual_seed(0)
    rule = lambda x: (x[:, 0] + x[:, 1] > 1).double()
    X, Xt = torch.rand(300, 2), torch.rand(500, 2)
    model, lik = SVGP(X[:40].clone()), BernoulliLikelihood()
    fit_variational(model, lik, X, rule(X), epochs=300)
    with torch.no_grad():
        acc = ((lik(model(Xt)).probs > 0.5).double() == rule(Xt)).double().mean().item()
    return {"accuracy": acc}


if __name__ == "__main__":
    print(main())
