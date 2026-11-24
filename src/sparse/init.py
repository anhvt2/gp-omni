"""Inducing-input initialisation.

``kmeans`` follows the landmark selection of the Improved Nystrom method (Zhang, Tsang &
Kwok, ICML 2008; K. Zhang's ``eff_kmeans.m``, BSD): a random subset of the data seeds a few
Lloyd iterations, and the cluster centres become the landmarks. Clusters that end up empty
are dropped, so fewer than ``nu`` points can come back (as in the original). The randomness
makes the result depend on ``seed``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("training")


def random_subset(x: Tensor, nu: int, seed: int = DEFAULTS["random_subset"]["seed"]) -> Tensor:
    """``nu`` training inputs chosen uniformly without replacement."""
    g = torch.Generator().manual_seed(seed)
    return x[torch.randperm(x.shape[0], generator=g)[:nu]].clone()


def kmeans_landmarks(
    x: Tensor,
    nu: int,
    max_iter: int = DEFAULTS["kmeans"]["max_iter"],
    seed: int = DEFAULTS["kmeans"]["seed"],
) -> Tensor:
    """Cluster centres after ``max_iter`` Lloyd iterations (Improved Nystrom landmarks)."""
    centre = random_subset(x, nu, seed)
    for _ in range(max_iter):
        idx = torch.cdist(centre, x).argmin(0)  # nearest centre of every data point
        keep = []
        for j in range(centre.shape[0]):
            members = x[idx == j]
            if members.shape[0] >= 1:
                centre[j] = members.mean(0)
                keep.append(j)
        centre = centre[keep]  # drop empty clusters
    return centre


def init_inducing(
    x: Tensor,
    nu: int,
    method: str = DEFAULTS["inducing"]["method"],
    seed: int = DEFAULTS["inducing"]["seed"],
) -> Tensor:
    if nu > x.shape[0]:
        raise ValueError("more inducing inputs than training points")
    if method == "kmeans":
        return kmeans_landmarks(x, nu, seed=seed)
    if method == "random":
        return random_subset(x, nu, seed)
    raise ValueError("method must be 'kmeans' or 'random'")
