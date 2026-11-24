"""Asynchronous parallel constrained GP-Hedge Bayesian optimisation.

Port of the master loop ``mainprog_benchR_gpml.m``. The loop never waits for a batch to
finish: whenever one of the ``exploit + explore + explore_classifier`` slots is free it fits the
surrogates on everything known so far, proposes one point and launches it, then goes back
for the next free slot. Jobs still running are *pending*: they sit in the database with
output 0 / feasible 0, so they are hallucinated with the objective model's own prediction
(kriging believer) and the next proposal is pushed away from them.

Each proposal maximises ``acquisition * known_constraint * P(feasible)`` (see
:mod:`gp_omni.asyncbo.acquisition`).

The maximisation convention is kept: to minimise, negate the objective.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from gp_omni.asyncbo import database as db
from gp_omni.asyncbo.acquisition import predict_objective
from gp_omni.asyncbo.sampling import (
    get_next_sampling_point,
    get_next_sampling_point_classifier_reduce_mse,
    get_next_sampling_point_reduce_mse,
)
from gp_omni.bo_config import BOConfig
from gp_omni.sparse.omni import OmniGP, OmniGPC

log = logging.getLogger(__name__)

Evaluator = Callable[[np.ndarray], tuple[float, bool]]  # x (unscaled) -> (objective, feasible)


@dataclass
class AsyncBOConfig(BOConfig):
    """Asynchronous GP-Hedge settings; ``max_iter`` counts evaluations launched."""


class AsyncBO:
    """Asynchronous parallel BO driven by the shared folder database.

    Provide either ``evaluate`` (a Python callable run in a thread pool) or
    ``config.query_script`` (a shell script run in each job folder that writes
    ``output.dat``, ``feasible.dat`` and finally ``complete.dat`` = 1, and detaches its own
    long-running work, exactly like the original ``query*.sh``).
    """

    def __init__(self, root: str | Path, config: BOConfig, evaluate: Evaluator | None = None):
        if (evaluate is None) == (config.query_script is None):
            raise ValueError("give exactly one of `evaluate` and `config.query_script`")
        self.root, self.config, self.evaluate = Path(root).resolve(), config, evaluate
        self.rng = np.random.default_rng(config.seed)
        self.pool = ThreadPoolExecutor(max_workers=config.batch.total) if evaluate else None
        self.futures: list[Future] = []
        self.model: OmniGP | None = None
        self.classifier: OmniGPC | None = None
        self.y_hallucinated: np.ndarray | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        db.write_settings(self.root, config.model_name, config.batch, config.wait_time)

    # ------------------------------------------------------------------ job launching
    def _folder(self, i: int) -> Path:
        return self.root / f"{self.config.model_name}_iter{i}"

    def _launch(
        self, folder: Path, x_scaled: np.ndarray, batch_id: int, scheme: str | None
    ) -> None:
        """Create the job folder (copy template, ``input.dat``, ...) and start the job."""
        folder.mkdir(exist_ok=True)
        template = self.root / f"{self.config.model_name}_template"
        if template.is_dir():
            shutil.copytree(template, folder, dirs_exist_ok=True)
        (folder / "batch_id.dat").write_text(f"{batch_id}\n")
        if scheme:
            (folder / "acquisition_scheme.dat").write_text(scheme)  # no trailing newline
        x = x_scaled * self.config.x_scale
        np.savetxt(folder / "input.dat", x.reshape(1, -1), fmt="%0.8e", delimiter=",")
        (folder / "complete.dat").write_text("0\n")  # 0 = queried, not finished
        if self.evaluate is not None:
            self.futures.append(self.pool.submit(self._run_job, folder, x))
        else:
            subprocess.run(["bash", str(self.config.query_script)], cwd=folder, check=False)

    def _run_job(self, folder: Path, x: np.ndarray) -> None:
        try:
            y, feasible = self.evaluate(x)
            feasible = bool(feasible) and bool(np.isfinite(y))
        except Exception:
            log.exception("evaluation failed in %s", folder.name)
            y, feasible = 0.0, False
        (folder / "output.dat").write_text(f"{y if feasible else 0.0:.8f}\n")
        (folder / "feasible.dat").write_text(f"{int(feasible)}\n")
        (folder / "complete.dat").write_text("1\n")  # last, so readers see a finished job

    # ------------------------------------------------------------------------ models
    def _load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_obs, y_obs, f_obs, _ = db.update_db(self.root)
        return x_obs / self.config.x_scale, y_obs / self.config.y_scale, f_obs

    def fit_models(self, x_obs: np.ndarray, y_obs: np.ndarray, f_obs: np.ndarray) -> None:
        """Hallucinate infeasible / pending outputs, then fit objective model and classifier.

        1. fit the objective on feasible points only; 2. replace the outputs of infeasible
        and pending points with that model's prediction; 3. refit on everything;
        4. fit the constraint classifier on ``f_obs`` (skipped while every label is the same).
        """
        n = self.config.gp_max_iter
        x_train = torch.as_tensor(x_obs)
        feasible = torch.as_tensor(f_obs > 0)
        self.model = self.classifier = self.y_hallucinated = None
        if feasible.sum() >= 2:
            interp = OmniGP().fit(x_train[feasible], torch.as_tensor(y_obs)[feasible], max_iter=n)
            y_obs = y_obs.copy()
            for i in np.flatnonzero(f_obs == 0):
                y_obs[i] = predict_objective(interp, x_obs[i])[0]
            self.model = OmniGP().fit(x_train, torch.as_tensor(y_obs), max_iter=n)
            self.y_hallucinated = y_obs
        if 0 < feasible.sum() < len(f_obs):
            self.classifier = OmniGPC().fit(x_train, torch.as_tensor(f_obs > 0), max_iter=n)

    # ------------------------------------------------------------------------ proposals
    def propose(self, batch_id: int, x_obs: np.ndarray) -> tuple[np.ndarray, str | None]:
        """One new point for slot kind ``batch_id`` (1 acquisition, 2 / 3 exploration)."""
        config, scheme = self.config, None
        if self.model is None:  # fewer than 2 feasible points: nothing to model yet
            return config.x_lb + self.rng.random(len(config.x_lb)) * (
                config.x_ub - config.x_lb
            ), scheme
        if batch_id == 1:
            rewards = db.get_rewards(self.root, config.num_init_point)
            scheme = db.get_acquisition_scheme(self.root, config.num_init_point, rewards, self.rng)
            x = get_next_sampling_point(
                self.model,
                self.classifier,
                scheme,
                config.x_lb,
                config.x_ub,
                x_obs,
                self.y_hallucinated,
                self.rng,
                config.known_constraint,
                config.cma_max_evals,
            )
        elif batch_id == 2:
            x = get_next_sampling_point_reduce_mse(self.model, config.x_lb, config.x_ub, self.rng)
        else:
            x = get_next_sampling_point_classifier_reduce_mse(
                self.classifier, config.x_lb, config.x_ub, self.rng
            )
        return x, scheme

    # ---------------------------------------------------------------------------- run
    def _seed_initial_points(self) -> None:
        """Evaluate ``num_init_point`` uniform random points (skipping existing folders)."""
        config = self.config
        for i in range(1, config.num_init_point + 1):
            if (self._folder(i) / "input.dat").is_file():
                continue
            x = config.x_lb + self.rng.random(len(config.x_lb)) * (config.x_ub - config.x_lb)
            self._launch(self._folder(i), x, 0, None)

    def _wait_for_initial(self) -> None:
        while True:
            db.check_complete(self.root)
            if all(
                db.read_folder(self._folder(i))[3] == 1
                for i in range(1, self.config.num_init_point + 1)
            ):
                return
            time.sleep(self.config.check_time * 60)

    def run(self) -> tuple[np.ndarray, float]:
        """Run to ``max_iter`` evaluations; return the best feasible ``(x, y)`` (unscaled)."""
        config = self.config
        self._seed_initial_points()
        self._wait_for_initial()
        x_obs, y_obs, f_obs = self._load()
        self.fit_models(x_obs, y_obs, f_obs)

        for i in range(config.num_init_point + 1, config.max_iter + 1):
            batch_id = db.wait_for_slot(self.root, config.check_time)  # blocks only if all busy
            log.info("iteration %d: batch %d, %d in flight", i, batch_id, self._in_flight())
            x, scheme = self.propose(batch_id, x_obs)
            folder = self._folder(i)
            if self.model is not None:  # gp_predictions.dat: predicted mean / variance, unscaled
                mu, s2 = predict_objective(self.model, x)
                folder.mkdir(exist_ok=True)
                (folder / "gp_predictions.dat").write_text(
                    f"{mu * config.y_scale:.8f}\n{s2 * config.y_scale**2:.8f}\n"
                )
            self._launch(folder, x, batch_id, scheme)
            x_obs, y_obs, f_obs = self._load()  # pick up whatever finished in the meantime
            self.fit_models(x_obs, y_obs, f_obs)

        self.drain()
        return self.best()

    def _in_flight(self) -> int:
        folders = db.iter_folders(self.root, self.config.model_name)
        return sum(db.read_folder(f)[3] == 0 for f in folders)

    def drain(self) -> None:
        """Wait for the evaluations still running and refresh the database."""
        for fut in self.futures:
            fut.result()
        if self.pool is not None:
            self.pool.shutdown()
        while self._in_flight():  # shell-script jobs finish on their own schedule
            db.check_complete(self.root)
            time.sleep(self.config.check_time * 60)
        db.update_db(self.root)

    def best(self) -> tuple[np.ndarray, float]:
        x_obs, y_obs, f_obs, _ = db.update_db(self.root)
        masked = np.where(f_obs > 0, y_obs, -np.inf)
        i = int(np.argmax(masked))
        return x_obs[i], float(masked[i])
