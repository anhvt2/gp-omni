"""Gaussian-process classification with composable covariance features."""

from __future__ import annotations

from torch import Tensor

from gp_omni.build.gpr import GPR
from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("models")


class GPC(GPR):
    """Constraint classifier: the Gaussian-process classifier twin of :class:`GPR`.

    Same constructor, same four independent switches, same ``fit`` / ``predict`` shape, so a
    classifier can mirror whatever the regression model uses: if the objectives are modelled
    with several correlated outputs, fidelities, an additive kernel and a sparse
    approximation, the constraints can be too, and each switch can be changed on its own.

    Labels are ``+1`` = feasible, ``-1`` = infeasible (also accepted: bool and ``{0, 1}``).
    The latent function ``f`` gets a probit (``probit``, default) or logistic likelihood
    and the posterior is the Laplace approximation (gpml ``infLaplace``) on the same exact or
    sparse covariance approximation as the regression twin. With ``num_outputs > 1``, each
    output is one constraint, correlated with the others; with ``num_fidelities > 1``, labels
    can come from cheap and expensive evaluations of the constraint.

    ``predict_proba`` returns ``P(feasible)``; ``predict`` also returns the latent mean
    ``fmu`` / variance ``fs2`` and ``ymu = 2 P - 1``.
    """

    def __init__(
        self,
        num_outputs: int = DEFAULTS["gpc"]["num_outputs"],
        num_fidelities: int = DEFAULTS["gpc"]["num_fidelities"],
        additive: int | list[int] | None = DEFAULTS["gpc"]["additive"],
        nu: int | None = DEFAULTS["gpc"]["nu"],
        s: float = DEFAULTS["gpc"]["s"],
        cov: str = DEFAULTS["gpc"]["cov"],
        degree: int = DEFAULTS["gpc"]["degree"],
        rank: int = DEFAULTS["gpc"]["rank"],
        lik: str = DEFAULTS["gpc"]["lik"],
        init: str = DEFAULTS["gpc"]["init"],
        seed: int = DEFAULTS["gpc"]["seed"],
    ):
        if lik == "likGauss":
            raise ValueError("GPC needs a classification likelihood: probit or logistic")
        super().__init__(
            num_outputs, num_fidelities, additive, nu, s, cov, degree, rank, lik, init, seed
        )

    def fit(
        self,
        x: Tensor,
        y: Tensor,
        output: Tensor | None = None,
        fidelity: Tensor | None = None,
        max_iter: int = DEFAULTS["gpc_fit"]["max_iter"],
        learn_xu: bool = DEFAULTS["gpc_fit"]["learn_xu"],
    ) -> GPC:
        """Fit on constraint labels ``y`` (``+1`` feasible); see :meth:`GPR.fit`."""
        from gp_omni.train.gpc import fit

        return fit(self, x, y, output, fidelity, max_iter, learn_xu)

    def predict_proba(
        self, xs: Tensor, output: int | None = None, fidelity: int | None = None
    ) -> Tensor:
        """``P(feasible)``, shape ``(m,)`` or ``(m, num_outputs)``."""
        from gp_omni.infer.gpc import predict_proba

        return predict_proba(self, xs, output, fidelity)

    def predict_feasible(
        self,
        xs: Tensor,
        output: int | None = None,
        fidelity: int | None = None,
        threshold: float = DEFAULTS["decision"]["threshold"],
    ) -> Tensor:
        """Boolean feasibility decision ``P(feasible) >= threshold``."""
        from gp_omni.infer.gpc import predict_feasible

        return predict_feasible(self, xs, output, fidelity, threshold)
