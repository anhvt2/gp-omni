"""Gaussian-process classification with composable covariance features."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.sparse.gpr import GPR


def _as_labels(y: Tensor) -> Tensor:
    """Constraint labels as floats in {-1, +1}: bool, {0, 1} or {-1, +1} are accepted."""
    y = torch.as_tensor(y)
    if y.dtype == torch.bool:
        return torch.where(y, 1.0, -1.0).to(torch.float64)
    y = y.to(torch.float64).reshape(-1)
    values = set(y.unique().tolist())
    if values <= {-1.0, 1.0}:
        return y
    if values <= {0.0, 1.0}:
        return 2.0 * y - 1.0
    raise ValueError("constraint labels must be bool, {0, 1} or {-1, +1}")


class GPC(GPR):
    """Constraint classifier: the Gaussian-process classifier twin of :class:`GPR`.

    Same constructor, same four independent switches, same ``fit`` / ``predict`` shape, so a
    classifier can mirror whatever the regression model uses: if the objectives are modelled
    with several correlated outputs, fidelities, an additive kernel and a sparse
    approximation, the constraints can be too, and each switch can be changed on its own.

    Labels are ``+1`` = feasible, ``-1`` = infeasible (also accepted: bool and ``{0, 1}``).
    The latent function ``f`` gets a probit (``likErf``, default) or logistic likelihood
    and the posterior is the Laplace approximation (gpml ``infLaplace``) on the same exact or
    sparse covariance approximation as the regression twin. With ``num_outputs > 1``, each
    output is one constraint, correlated with the others; with ``num_fidelities > 1``, labels
    can come from cheap and expensive evaluations of the constraint.

    ``predict_proba`` returns ``P(feasible)``; ``predict`` also returns the latent mean
    ``fmu`` / variance ``fs2`` and ``ymu = 2 P - 1``.
    """

    def __init__(
        self,
        num_outputs: int = 1,
        num_fidelities: int = 1,
        additive: int | list[int] | None = None,
        nu: int | None = None,
        s: float = 0.0,
        cov: str = "SEard",
        degree: int = 3,
        rank: int = 1,
        lik: str = "likErf",
        init: str = "kmeans",
        seed: int = 0,
    ):
        if lik == "likGauss":
            raise ValueError("GPC needs a classification likelihood: likErf or likLogistic")
        super().__init__(
            num_outputs, num_fidelities, additive, nu, s, cov, degree, rank, lik, init, seed
        )

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        output: Tensor | None = None,
        fidelity: Tensor | None = None,
        max_iter: int = 200,
        learn_xu: bool = True,
    ) -> GPC:
        """Fit on constraint labels ``y`` (``+1`` feasible); see :meth:`GPR.fit`."""
        super().fit(x, _as_labels(y), output, fidelity, max_iter, learn_xu)
        return self

    def predict_proba(
        self, xs: Tensor, output: int | None = None, fidelity: int | None = None
    ) -> Tensor:
        """``P(feasible)``, shape ``(m,)`` or ``(m, num_outputs)``."""
        return self.predict(xs, output, fidelity)["prob"]

    def predict_feasible(
        self,
        xs: Tensor,
        output: int | None = None,
        fidelity: int | None = None,
        threshold: float = 0.5,
    ) -> Tensor:
        """Boolean feasibility decision ``P(feasible) >= threshold``."""
        return self.predict_proba(xs, output, fidelity) >= threshold
