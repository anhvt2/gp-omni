"""GP-Hedge portfolio of acquisition functions for the asynchronous loop.

Hoffman, Brochu & de Freitas, *Portfolio Allocation for Bayesian Optimization*, UAI 2011.
The arms are any names of :mod:`gp_omni.opt.acquisition`; ``rewards.dat`` has one line per
arm, in the order given.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gp_omni.opt.parallel.jobs import _num, iter_folders, read_settings

log = logging.getLogger(__name__)

DEFAULT_ARMS = ("UCB", "EI", "PI")  # order of the lines in rewards.dat


def get_rewards(
    root: Path, num_init_point: int, arms: tuple[str, ...] = DEFAULT_ARMS
) -> np.ndarray:
    """``getRewards.py``: GP-Hedge rewards, one per arm, written to ``rewards.dat``.

    An acquisition function earns +1 each time one of its points set a new best-so-far
    objective (infeasible / unfinished points, whose output is 0 or missing, never count).
    """
    model_name, _, _ = read_settings(root)
    ys, acquisitions = [], []
    for folder in iter_folders(root, model_name):
        output = folder / "output.dat"
        ys.append(_num(output, np.nan))
        scheme = folder / "acquisition_scheme.dat"
        acquisitions.append(scheme.read_text().strip() if scheme.is_file() else None)

    best, winners = ys[0], []
    for i in range(num_init_point, len(ys)):
        if best < ys[i] and ys[i] != 0:  # NaN (unfinished) compares False
            winners.append(acquisitions[i])
            best = ys[i]
    rewards = np.array([float(winners.count(name)) for name in arms])
    (root / "rewards.dat").write_text("".join(f"{r:.2f}\n" for r in rewards))
    return rewards


def get_acquisition_scheme(
    root: Path,
    num_init_point: int,
    rewards: np.ndarray,
    rng: np.random.Generator,
    arms: tuple[str, ...] = DEFAULT_ARMS,
) -> str:
    """``getAcquisitionScheme.py``: sample one arm from the GP-Hedge softmax."""
    model_name, _, _ = read_settings(root)
    horizon = max(len(iter_folders(root, model_name)) - (num_init_point - 1), 1)
    k = len(arms)
    eta = np.sqrt(8 * np.log(k) / horizon)  # lemma 1, GP-Hedge
    shifted = rewards - rewards.max()  # numerical stability: exp of non-positive numbers only
    pmf = np.exp(eta * shifted) / np.exp(eta * shifted).sum()
    scheme = str(rng.choice(arms, p=pmf))
    log.info("%s is selected (pmf=%s)", scheme, np.round(pmf, 3))
    return scheme
