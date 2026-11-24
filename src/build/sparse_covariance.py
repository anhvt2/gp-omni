"""Sparse covariance definition and inducing-input lookup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from gp_omni.build.covariance import Cov


@dataclass
class SparseCov:
    """gpml's ``{@apxSparse, cov, xu}``: a covariance plus (default) inducing inputs.

    As in gpml, a ``hyp['xu']`` passed to inference takes priority over ``xu``.
    """

    cov: Cov
    xu: Tensor | None = None
    expand: Callable[[Tensor], Tensor] | None = None
    """Optional map from the stored inducing inputs to the inputs the covariance sees, e.g.
    replicating each location once per task (see ``multitask.task_expand``)."""


def get_xu(hyp: dict, sc: SparseCov) -> Tensor:
    """The inducing inputs used by the covariance: ``hyp['xu']`` has priority (gpml)."""
    xu = hyp.get("xu", sc.xu)
    if xu is None:
        raise ValueError("no inducing inputs: set sparse_covariance.xu or hyp['xu']")
    return sc.expand(xu) if sc.expand is not None else xu
