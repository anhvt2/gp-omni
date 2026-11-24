"""Batch-sequential parallel constrained Bayesian optimisation.

Port of the master loop ``mainprog_benchR.m``. Every iteration proposes a whole batch of
points, runs the batch in parallel, and moves on only when the *entire* batch has finished
(the asynchronous variant in :mod:`gp_omni.asyncbo` instead refills slots one by one).

Inside an iteration the points are chosen one after another with hallucination: after each
proposal the point and the objective model's own prediction there are added to the data
(labelled feasible) and both models are refitted, which pushes the next proposal elsewhere.
The batch is filled in three parts, each launched as soon as its point is chosen:

1. ``exploit_size`` points maximising ``EI * known_constraint * P(feasible)``;
2. ``explore_size`` points of maximal objective-model uncertainty;
3. ``explore_classifier_size`` points of maximal constraint-classifier uncertainty.

After the batch completes, the real results are collected (infeasible outputs are
hallucinated from the feasible ones) and the models are refitted for the next iteration.

``max_iter`` counts iterations as in the MATLAB loop: the initial points are iterations
``1..num_init_point`` and every later iteration launches one batch of
``exploit_size + explore_size + explore_classifier_size`` jobs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from gp_omni.asyncbo import database as db
from gp_omni.asyncbo.acquisition import predict_objective
from gp_omni.asyncbo.loop import AsyncBO
from gp_omni.asyncbo.sampling import (
    get_next_sampling_point,
    get_next_sampling_point_classifier_reduce_mse,
    get_next_sampling_point_reduce_mse,
)
from gp_omni.bo_config import BOConfig
from gp_omni.build.gpc import GPC
from gp_omni.build.gpr import GPR
from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("bo")

log = logging.getLogger(__name__)

KINDS = (("exploit", 1), ("explore", 2), ("explore_classifier", 3))  # (folder name, batch id)


@dataclass
class BatchSeqBOConfig(BOConfig):
    """Batch-sequential settings; ``max_iter`` counts iterations.

    ``check_time`` is the polling interval (minutes) while waiting for a batch to finish.
    """

    acquisition: str = DEFAULTS["batch"][
        "acquisition"
    ]  # PI, EI or UCB for the exploitation part of the batch
    relax_alpha: float | None = DEFAULTS["batch"][
        "relax_alpha"
    ]  # classifier probabilities rounded outside [1-a, a]; None=off


class BatchSeqBO(AsyncBO):
    """Batch-sequential parallel BO; shares job launching and the folder protocol with
    :class:`~gp_omni.asyncbo.AsyncBO` (``<model>_iter<i>_<kind>_batch<j>`` per job)."""

    config: BatchSeqBOConfig

    def _batch_folder(self, i: int, kind: str, j: int):
        return self.root / f"{self.config.model_name}_iter{i}_{kind}_batch{j}"

    # ------------------------------------------------------------- hallucination
    def _hallucinate(self, x_obs, y_obs, f_obs, x_new):
        """Append ``x_new`` with the objective model's mean, labelled feasible, and refit."""
        mu, _ = predict_objective(self.model, x_new)
        x_obs = np.vstack([x_obs, x_new])
        y_obs = np.append(y_obs, mu)
        f_obs = np.append(f_obs, 1.0)
        n = self.config.gp_max_iter
        self.model = GPR().fit(torch.as_tensor(x_obs), torch.as_tensor(y_obs), max_iter=n)
        self.classifier = None
        if (f_obs == 0).any():
            self.classifier = GPC().fit(
                torch.as_tensor(x_obs), torch.as_tensor(f_obs > 0), max_iter=n
            )
        self.y_hallucinated = y_obs
        return x_obs, y_obs, f_obs

    def _propose(self, batch_id: int, x_obs: np.ndarray) -> np.ndarray:
        c = self.config
        if batch_id == 1:
            return get_next_sampling_point(
                self.model, self.classifier, c.acquisition, c.x_lb, c.x_ub, x_obs,
                self.y_hallucinated, self.rng, c.known_constraint, c.cma_max_evals,
                relax_alpha=c.relax_alpha,
            )  # fmt: skip
        if batch_id == 2:
            return get_next_sampling_point_reduce_mse(self.model, c.x_lb, c.x_ub, self.rng)
        return get_next_sampling_point_classifier_reduce_mse(
            self.classifier, c.x_lb, c.x_ub, self.rng
        )

    def launch_batch(self, i: int, x_obs, y_obs, f_obs) -> list:
        """Choose and launch the whole batch of iteration ``i`` with hallucination."""
        c = self.config
        folders = []
        sizes = {"exploit": c.exploit_size, "explore": c.explore_size}
        sizes["explore_classifier"] = c.explore_classifier_size
        for kind, batch_id in KINDS:
            for j in range(1, sizes[kind] + 1):
                if self.model is None:  # fewer than 2 feasible points: nothing to model yet
                    x = c.x_lb + self.rng.random(len(c.x_lb)) * (c.x_ub - c.x_lb)
                else:
                    x = self._propose(batch_id, x_obs)
                folder = self._batch_folder(i, kind, j)
                scheme = c.acquisition if batch_id == 1 else None
                self._launch(folder, x, batch_id, scheme)  # starts running immediately
                folders.append(folder)
                if self.model is not None:
                    x_obs, y_obs, f_obs = self._hallucinate(x_obs, y_obs, f_obs, x)
        return folders

    def wait_for_batch(self, folders: list) -> None:
        """Block until every job of the batch is complete (``checkComplete.py``)."""
        while True:
            db.check_complete(self.root)
            if all(db.read_folder(f)[3] == 1 for f in folders):
                return
            time.sleep(self.config.check_time * 60)

    # ---------------------------------------------------------------------------- run
    def run(self) -> tuple[np.ndarray, float]:
        """Run iterations ``num_init_point + 1 .. max_iter``; return the best feasible
        ``(x, y)`` (unscaled)."""
        c = self.config
        self._seed_initial_points()
        self._wait_for_initial()
        x_obs, y_obs, f_obs = self._load()
        self.fit_models(x_obs, y_obs, f_obs)

        for i in range(c.num_init_point + 1, c.max_iter + 1):
            log.info("iteration %d: launching a batch of %d jobs", i, c.batch.total)
            y_start = self.y_hallucinated if self.y_hallucinated is not None else y_obs
            folders = self.launch_batch(i, x_obs, y_start, f_obs)
            self.wait_for_batch(folders)
            x_obs, y_obs, f_obs = self._load()  # real results of the finished batch
            self.fit_models(x_obs, y_obs, f_obs)

        self.drain()
        return self.best()
