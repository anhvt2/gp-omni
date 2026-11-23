"""File-based asynchronous job pool: port of the ``bayesOptSrc-gpml/*.py`` helper scripts.

Layout (unchanged from the MATLAB/Python original so existing ``query*.sh`` scripts and
``<model>_Template/`` folders keep working)::

    root/
      batchSettings.dat  waitTime.dat  modelName.dat
      S.dat Y.dat F.dat C.dat B.dat            global database (written by ``update_db``)
      <model>_Iter<i>/
        input.dat output.dat feasible.dat complete.dat batchID.dat
        acquisitionScheme.dat gpPredictions.dat   per-evaluation files

A job is *in flight* while its ``complete.dat`` is 0. Batch IDs: 0 initial, 1 acquisition
(GP-Hedge), 2 explore objective, 3 explore classifier, 4 all slots busy.
"""

from __future__ import annotations

import datetime
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from natsort import natsorted

log = logging.getLogger(__name__)

ACQUISITIONS = ("UCB", "EI", "PI")  # order of the lines in R.dat


@dataclass(frozen=True)
class BatchSettings:
    """``batchSettings.dat``: how many jobs of each kind may be in flight."""

    exploit: int
    explore: int
    explore_clf: int

    @property
    def total(self) -> int:
        return self.exploit + self.explore + self.explore_clf


def _num(path: Path, default: float) -> float:
    """Last number in ``path``, or ``default`` if the file does not exist / is empty."""
    if not path.is_file():
        return default
    values = np.atleast_1d(np.loadtxt(path, ndmin=1))
    return float(values[-1]) if values.size else default


def iter_folders(root: Path, model_name: str) -> list[Path]:
    """``<model>_Iter*`` folders in natural order."""
    return [
        Path(p) for p in natsorted(str(p) for p in root.glob(f"{model_name}_Iter*") if p.is_dir())
    ]


def write_settings(root: Path, model_name: str, batch: BatchSettings, wait_time: float) -> None:
    (root / "batchSettings.dat").write_text(
        f"{batch.exploit}\n{batch.explore}\n{batch.explore_clf}\n"
    )
    (root / "waitTime.dat").write_text(f"{wait_time:.4f}\n")
    (root / "modelName.dat").write_text(f"{model_name}\n")


def read_settings(root: Path) -> tuple[str, BatchSettings, float]:
    model_name = (root / "modelName.dat").read_text().strip()
    exploit, explore, explore_clf = (
        int(v) for v in np.loadtxt(root / "batchSettings.dat", dtype=int)
    )
    return (
        model_name,
        BatchSettings(exploit, explore, explore_clf),
        float(np.loadtxt(root / "waitTime.dat")),
    )


def read_folder(folder: Path) -> tuple[np.ndarray, float, float, float]:
    """``(x, y, feasible, complete)`` of one job; missing results count as 0 (unfinished)."""
    x = np.atleast_1d(np.loadtxt(folder / "input.dat", delimiter=",", ndmin=1))
    return (
        x,
        _num(folder / "output.dat", 0.0),
        _num(folder / "feasible.dat", 0.0),
        _num(folder / "complete.dat", 0.0),
    )


def check_complete(root: Path) -> None:
    """``checkComplete.py``: declare jobs that outlived ``waitTime`` complete and infeasible."""
    wait_hours = float(np.loadtxt(root / "waitTime.dat"))
    for complete_file in root.glob("*/complete.dat"):
        folder = complete_file.parent
        timeout_log = folder / "log.timeout"
        if _num(complete_file, 0.0) == 0 and timeout_log.is_file():
            age = datetime.datetime.now() - datetime.datetime.fromtimestamp(
                timeout_log.stat().st_mtime
            )
            if age.total_seconds() > wait_hours * 3600:
                log.warning("%s timed out; marking complete and infeasible", folder.name)
                (folder / "complete.dat").write_text("1\n")
                (folder / "output.dat").write_text("0\n")
                (folder / "feasible.dat").write_text("0\n")


def get_batch(root: Path) -> int:
    """``getBatch.py``: which kind of job to launch next; 4 when every slot is busy.

    Only in-flight jobs (``complete.dat`` = 0) occupy a slot; kinds are filled in priority
    order acquisition -> explore -> explore-classifier.
    """
    model_name, batch, _ = read_settings(root)
    busy = [
        _num(f / "batchID.dat", 0)
        for f in iter_folders(root, model_name)
        if _num(f / "complete.dat", 0) == 0
    ]
    for batch_id, limit in ((1, batch.exploit), (2, batch.explore), (3, batch.explore_clf)):
        if busy.count(batch_id) < limit:
            return batch_id
    return 4


def update_db(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``updateDb.py``: gather every job into ``S, Y, F, C`` and write them to ``*.dat``."""
    model_name, _, _ = read_settings(root)
    rows = [read_folder(f) for f in iter_folders(root, model_name)]
    S = np.vstack([r[0] for r in rows])
    Y, F, C = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    np.savetxt(root / "S.dat", S, fmt="%.16f", delimiter=",")
    np.savetxt(root / "Y.dat", Y, fmt="%.16f", delimiter=",")
    np.savetxt(root / "F.dat", F, fmt="%d", delimiter=",")
    np.savetxt(root / "C.dat", C, fmt="%d", delimiter=",")
    return S, Y, F, C


def get_rewards(root: Path, num_init_point: int) -> np.ndarray:
    """``getRewards.py``: GP-Hedge rewards ``[UCB, EI, PI]`` written to ``R.dat``.

    An acquisition function earns +1 each time one of its points set a new best-so-far
    objective (infeasible / unfinished points, whose output is 0 or missing, never count).
    """
    model_name, _, _ = read_settings(root)
    ys, acquisitions = [], []
    for folder in iter_folders(root, model_name):
        output = folder / "output.dat"
        ys.append(_num(output, np.nan))
        scheme = folder / "acquisitionScheme.dat"
        acquisitions.append(scheme.read_text().strip() if scheme.is_file() else None)

    best, winners = ys[0], []
    for i in range(num_init_point, len(ys)):
        if best < ys[i] and ys[i] != 0:  # NaN (unfinished) compares False
            winners.append(acquisitions[i])
            best = ys[i]
    rewards = np.array([float(winners.count(name)) for name in ACQUISITIONS])
    (root / "R.dat").write_text("".join(f"{r:.2f}\n" for r in rewards))
    return rewards


def get_acquisition_scheme(
    root: Path, num_init_point: int, rewards: np.ndarray, rng: np.random.Generator
) -> str:
    """``getAcquisitionScheme.py``: sample UCB / EI / PI from the GP-Hedge softmax.

    Hoffman, Brochu & de Freitas, *Portfolio Allocation for Bayesian Optimization*, UAI 2011.
    """
    model_name, _, _ = read_settings(root)
    T = max(len(iter_folders(root, model_name)) - (num_init_point - 1), 1)
    k = len(ACQUISITIONS)
    eta = np.sqrt(8 * np.log(k) / T)  # lemma 1, GP-Hedge
    R = rewards - rewards.max()  # numerical stability: exp of non-positive numbers only
    pmf = np.exp(eta * R) / np.exp(eta * R).sum()
    scheme = str(rng.choice(ACQUISITIONS, p=pmf))
    log.info("%s is selected (pmf=%s)", scheme, np.round(pmf, 3))
    return scheme


def wait_for_slot(root: Path, check_time_minutes: float) -> int:
    """Poll until ``get_batch`` returns a free slot (the master loop's ``while batchID == 4``)."""
    while True:
        check_complete(root)
        batch_id = get_batch(root)
        if batch_id != 4:
            return batch_id
        time.sleep(check_time_minutes * 60)
