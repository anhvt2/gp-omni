"""gpml-style names -> GPyTorch objects.

For MATLAB users: ``gpml_kernel("covSEard", dim)`` returns what ``{@covSEard}`` means in gpml
(``sf^2 * exp(-r^2/2)`` with per-dimension length scales). The full mapping is in
``docs/gpml-mapping.md``. Hyperparameters are *not* log-transformed the way gpml's ``hyp``
vectors are; GPyTorch owns the parameterisation.
"""

from __future__ import annotations

from collections.abc import Sequence

from gpytorch.kernels import (
    Kernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PolynomialKernel,
    ProductKernel,
    RBFKernel,
    RQKernel,
    ScaleKernel,
)
from gpytorch.likelihoods import BernoulliLikelihood, GaussianLikelihood, Likelihood
from gpytorch.means import ConstantMean, LinearMean, Mean, ZeroMean

_STATIONARY = {"SE": RBFKernel, "RQ": RQKernel, "PER": PeriodicKernel}


def gpml_kernel(spec: str | tuple | Sequence, dim: int) -> Kernel:
    """Build a kernel from a gpml-style spec.

    Examples: ``"covSEiso"``, ``"covSEard"``, ``("covMaternard", 5)``, ``("covPoly", 2)``,
    ``("covSum", ["covSEard", "covLINard"])``, ``("covProd", [...])``.
    """
    name, arg = (spec, None) if isinstance(spec, str) else (spec[0], spec[1])
    if name in ("covSum", "covProd"):
        assert arg is not None, f"{name} needs a list of covariances"
        parts = [gpml_kernel(s, dim) for s in arg]
        out = parts[0]
        for p in parts[1:]:
            out = out + p if name == "covSum" else out * p
        return out
    if not name.startswith("cov"):
        raise ValueError(f"not a gpml covariance name: {name!r}")
    body = name[3:]
    ard = body.endswith("ard")
    body = body.removesuffix("ard").removesuffix("iso")
    ard_dims = dim if ard else None
    if body == "Matern":
        d = 3 if arg is None else int(arg)
        if d not in (1, 3, 5):
            raise ValueError("gpml Matern degree must be 1, 3 or 5")
        return ScaleKernel(MaternKernel(nu=d / 2, ard_num_dims=ard_dims))
    if body in _STATIONARY:
        return ScaleKernel(_STATIONARY[body](ard_num_dims=ard_dims))
    if body == "LIN":
        return ScaleKernel(LinearKernel())
    if body == "Poly":
        return ScaleKernel(PolynomialKernel(power=int(arg or 2)))
    raise ValueError(f"unsupported gpml covariance {name!r}")


def gpml_mean(name: str, dim: int) -> Mean:
    """``meanZero`` / ``meanConst`` / ``meanLinear`` (``meanSum`` of linear+const: use LinearMean)."""
    table = {
        "meanZero": ZeroMean(),
        "meanConst": ConstantMean(),
        "meanLinear": LinearMean(dim, bias=False),
        "meanSum": LinearMean(dim, bias=True),  # gpml {@meanSum,{@meanLinear,@meanConst}}
    }
    if name not in table:
        raise ValueError(f"unsupported gpml mean {name!r}")
    return table[name]


def gpml_likelihood(name: str) -> Likelihood:
    """``likGauss`` -> Gaussian noise; ``likErf`` -> probit Bernoulli (variational only)."""
    if name == "likGauss":
        return GaussianLikelihood()
    if name == "likErf":
        return BernoulliLikelihood()
    raise ValueError(f"unsupported gpml likelihood {name!r}")


__all__ = ["ProductKernel", "gpml_kernel", "gpml_likelihood", "gpml_mean"]
