"""Sparse covariance definition and inducing-input lookup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from gp_omni.build.covariance import covariance


@dataclass
class sparse_covariance:
    """A covariance plus (default) inducing inputs.

    A ``hyp['xu']`` passed to inference takes priority over ``xu``.
    """

    cov: covariance
    xu: Tensor | None = None
    expand: Callable[[Tensor], Tensor] | None = None
    """Optional map from the stored inducing inputs to the inputs the covariance sees, e.g.
    replicating each location once per task (see ``multitask.task_expand``)."""


def get_xu(hyp: dict, sc: sparse_covariance) -> Tensor:
    """The inducing inputs used by the covariance: ``hyp['xu']`` has priority."""
    xu = hyp.get("xu", sc.xu)
    if xu is None:
        raise ValueError("no inducing inputs: set sparse_covariance.xu or hyp['xu']")
    return sc.expand(xu) if sc.expand is not None else xu
