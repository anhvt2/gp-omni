"""Asynchronous parallel constrained GP-Hedge Bayesian optimisation.

Port of the master loop ``mainprog_benchR_gpml.m``. The loop never waits for a batch to
finish: whenever one of the ``exploit + explore + explore_clf`` slots is free it fits the
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from gp_omni.asyncbo import database as db
from gp_omni.asyncbo.acquisition import KnownConstraint, check_constraint, predict_objective
from gp_omni.asyncbo.sampling import (
    get_next_sampling_point,
    get_next_sampling_point_clf_reduce_mse,
    get_next_sampling_point_reduce_mse,
)
from gp_omni.sparse.omni import OmniGP, OmniGPC

log = logging.getLogger(__name__)

Evaluator = Callable[[np.ndarray], tuple[float, bool]]  # x (unscaled) -> (objective, feasible)


@dataclass
class AsyncBOConfig:
    """Settings of ``mainprog_benchR_gpml.m``."""

    model_name: str
    x_lb: np.ndarray
    x_ub: np.ndarray
    exploit_size: int = 3  # concurrent acquisition (GP-Hedge) jobs
    explore_size: int = 2  # concurrent max-variance jobs on the objective model
    explore_clf_size: int = 1  # concurrent max-variance jobs on the constraint classifier
    max_iter: int = 400  # total number of evaluations launched, initial ones included
    num_init_point: int = 2
    x_scale: np.ndarray | float = 1.0  # rough scale so scaled inputs are O(1)
    y_scale: float = 1.0
    check_time: float = 0.10  # minutes between polls while every slot is busy
    wait_time: float = 1.0  # hours before an unfinished job is declared failed
    gp_max_iter: int = 200  # hyper-parameter optimiser iterations per fit
    cma_max_evals: int = 250  # CMA-ES budget per acquisition proposal (explore uses 100)
    seed: int | None = None
    query_script: str | None = None  # bash script run inside the job folder (async by itself)
    known_constraint: KnownConstraint = field(default=check_constraint)

    def __post_init__(self):
        self.x_lb, self.x_ub = np.asarray(self.x_lb, float), np.asarray(self.x_ub, float)
        self.x_scale = np.broadcast_to(np.asarray(self.x_scale, float), self.x_lb.shape).copy()

    @property
    def batch(self) -> db.BatchSettings:
        return db.BatchSettings(self.exploit_size, self.explore_size, self.explore_clf_size)


class AsyncBO:
    """Asynchronous parallel BO driven by the shared folder database.

    Provide either ``evaluate`` (a Python callable run in a thread pool) or
    ``config.query_script`` (a shell script run in each job folder that writes
    ``output.dat``, ``feasible.dat`` and finally ``complete.dat`` = 1, and detaches its own
    long-running work, exactly like the original ``query*.sh``).
    """

    def __init__(self, root: str | Path, config: AsyncBOConfig, evaluate: Evaluator | None = None):
        if (evaluate is None) == (config.query_script is None):
            raise ValueError("give exactly one of `evaluate` and `config.query_script`")
        self.root, self.cfg, self.evaluate = Path(root).resolve(), config, evaluate
        self.rng = np.random.default_rng(config.seed)
        self.pool = ThreadPoolExecutor(max_workers=config.batch.total) if evaluate else None
        self.futures: list[Future] = []
        self.model: OmniGP | None = None
        self.clf: OmniGPC | None = None
        self.y_halluc: np.ndarray | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        db.write_settings(self.root, config.model_name, config.batch, config.wait_time)

    # ------------------------------------------------------------------ job launching
    def _folder(self, i: int) -> Path:
        return self.root / f"{self.cfg.model_name}_Iter{i}"

    def _launch(
        self, folder: Path, x_scaled: np.ndarray, batch_id: int, scheme: str | None
    ) -> None:
        """Create the job folder (copy template, ``input.dat``, ...) and start the job."""
        folder.mkdir(exist_ok=True)
        template = self.root / f"{self.cfg.model_name}_Template"
        if template.is_dir():
            shutil.copytree(template, folder, dirs_exist_ok=True)
        (folder / "batchID.dat").write_text(f"{batch_id}\n")
        if scheme:
            (folder / "acquisitionScheme.dat").write_text(scheme)  # no trailing newline
        x = x_scaled * self.cfg.x_scale
        np.savetxt(folder / "input.dat", x.reshape(1, -1), fmt="%0.8e", delimiter=",")
        (folder / "complete.dat").write_text("0\n")  # 0 = queried, not finished
        if self.evaluate is not None:
            self.futures.append(self.pool.submit(self._run_job, folder, x))
        else:
            subprocess.run(["bash", str(self.cfg.query_script)], cwd=folder, check=False)

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
        return x_obs / self.cfg.x_scale, y_obs / self.cfg.y_scale, f_obs

    def fit_models(self, x_obs: np.ndarray, y_obs: np.ndarray, f_obs: np.ndarray) -> None:
        """Hallucinate infeasible / pending outputs, then fit objective model and classifier.

        1. fit the objective on feasible points only; 2. replace the outputs of infeasible
        and pending points with that model's prediction; 3. refit on everything;
        4. fit the constraint classifier on ``f_obs`` (skipped while every label is the same).
        """
        n = self.cfg.gp_max_iter
        x_train = torch.as_tensor(x_obs)
        feasible = torch.as_tensor(f_obs > 0)
        self.model = self.clf = self.y_halluc = None
        if feasible.sum() >= 2:
            interp = OmniGP().fit(x_train[feasible], torch.as_tensor(y_obs)[feasible], max_iter=n)
            y_obs = y_obs.copy()
            for i in np.flatnonzero(f_obs == 0):
                y_obs[i] = predict_objective(interp, x_obs[i])[0]
            self.model = OmniGP().fit(x_train, torch.as_tensor(y_obs), max_iter=n)
            self.y_halluc = y_obs
        if 0 < feasible.sum() < len(f_obs):
            self.clf = OmniGPC().fit(x_train, torch.as_tensor(f_obs > 0), max_iter=n)

    # ------------------------------------------------------------------------ proposals
    def propose(self, batch_id: int, x_obs: np.ndarray) -> tuple[np.ndarray, str | None]:
        """One new point for slot kind ``batch_id`` (1 acquisition, 2 / 3 exploration)."""
        cfg, scheme = self.cfg, None
        if self.model is None:  # fewer than 2 feasible points: nothing to model yet
            return cfg.x_lb + self.rng.random(len(cfg.x_lb)) * (cfg.x_ub - cfg.x_lb), scheme
        if batch_id == 1:
            rewards = db.get_rewards(self.root, cfg.num_init_point)
            scheme = db.get_acquisition_scheme(self.root, cfg.num_init_point, rewards, self.rng)
            x = get_next_sampling_point(
                self.model,
                self.clf,
                scheme,
                cfg.x_lb,
                cfg.x_ub,
                x_obs,
                self.y_halluc,
                self.rng,
                cfg.known_constraint,
                cfg.cma_max_evals,
            )
        elif batch_id == 2:
            x = get_next_sampling_point_reduce_mse(self.model, cfg.x_lb, cfg.x_ub, self.rng)
        else:
            x = get_next_sampling_point_clf_reduce_mse(self.clf, cfg.x_lb, cfg.x_ub, self.rng)
        return x, scheme

    # ---------------------------------------------------------------------------- run
    def _seed_initial_points(self) -> None:
        """Evaluate ``num_init_point`` uniform random points (skipping existing folders)."""
        cfg = self.cfg
        for i in range(1, cfg.num_init_point + 1):
            if (self._folder(i) / "input.dat").is_file():
                continue
            x = cfg.x_lb + self.rng.random(len(cfg.x_lb)) * (cfg.x_ub - cfg.x_lb)
            self._launch(self._folder(i), x, 0, None)

    def _wait_for_initial(self) -> None:
        while True:
            db.check_complete(self.root)
            if all(
                db.read_folder(self._folder(i))[3] == 1
                for i in range(1, self.cfg.num_init_point + 1)
            ):
                return
            time.sleep(self.cfg.check_time * 60)

    def run(self) -> tuple[np.ndarray, float]:
        """Run to ``max_iter`` evaluations; return the best feasible ``(x, y)`` (unscaled)."""
        cfg = self.cfg
        self._seed_initial_points()
        self._wait_for_initial()
        x_obs, y_obs, f_obs = self._load()
        self.fit_models(x_obs, y_obs, f_obs)

        for i in range(cfg.num_init_point + 1, cfg.max_iter + 1):
            batch_id = db.wait_for_slot(self.root, cfg.check_time)  # blocks only if all busy
            log.info("iteration %d: batch %d, %d in flight", i, batch_id, self._in_flight())
            x, scheme = self.propose(batch_id, x_obs)
            folder = self._folder(i)
            if self.model is not None:  # gpPredictions.dat: predicted mean / variance, unscaled
                mu, s2 = predict_objective(self.model, x)
                folder.mkdir(exist_ok=True)
                (folder / "gpPredictions.dat").write_text(
                    f"{mu * cfg.y_scale:.8f}\n{s2 * cfg.y_scale**2:.8f}\n"
                )
            self._launch(folder, x, batch_id, scheme)
            x_obs, y_obs, f_obs = self._load()  # pick up whatever finished in the meantime
            self.fit_models(x_obs, y_obs, f_obs)

        self.drain()
        return self.best()

    def _in_flight(self) -> int:
        folders = db.iter_folders(self.root, self.cfg.model_name)
        return sum(db.read_folder(f)[3] == 0 for f in folders)

    def drain(self) -> None:
        """Wait for the evaluations still running and refresh the database."""
        for fut in self.futures:
            fut.result()
        if self.pool is not None:
            self.pool.shutdown()
        db.update_db(self.root)

    def best(self) -> tuple[np.ndarray, float]:
        x_obs, y_obs, f_obs, _ = db.update_db(self.root)
        masked = np.where(f_obs > 0, y_obs, -np.inf)
        i = int(np.argmax(masked))
        return x_obs[i], float(masked[i])
