"""File-based job pool shared by the batch-sequential and asynchronous loops.

GP-Hedge bookkeeping (``rewards.dat``) lives in :mod:`gp_omni.opt.parallel.asyncbo.hedge`.


Layout (kept stable so existing ``query*.sh`` scripts and
``<model>_template/`` folders keep working)::

    root/
      batch_settings.dat  wait_time.dat  model_name.dat
      S.dat Y.dat F.dat C.dat B.dat            global database (written by ``update_db``)
      <model>_iter<i>/
        input.dat output.dat feasible.dat complete.dat batch_id.dat
        acquisition_scheme.dat gp_predictions.dat   per-evaluation files

A job is *in flight* while its ``complete.dat`` is 0. Batch IDs: 0 initial, 1 acquisition
(acquisition; GP-Hedge in async runs), 2 explore objective, 3 explore classifier, 4 all slots busy.
"""

from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import numpy as np
from natsort import natsorted

from gp_omni.opt.config import batch_settings

log = logging.getLogger(__name__)


def _num(path: Path, default: float) -> float:
    """Last number in ``path``, or ``default`` if the file does not exist / is empty."""
    if not path.is_file():
        return default
    values = np.atleast_1d(np.loadtxt(path, ndmin=1))
    return float(values[-1]) if values.size else default


def iter_folders(root: Path, model_name: str) -> list[Path]:
    """``<model>_iter*`` folders in natural order."""
    return [
        Path(p) for p in natsorted(str(p) for p in root.glob(f"{model_name}_iter*") if p.is_dir())
    ]


def write_settings(root: Path, model_name: str, batch: batch_settings, wait_time: float) -> None:
    (root / "batch_settings.dat").write_text(
        f"{batch.exploit}\n{batch.explore}\n{batch.explore_classifier}\n"
    )
    (root / "wait_time.dat").write_text(f"{wait_time:.4f}\n")
    (root / "model_name.dat").write_text(f"{model_name}\n")


def read_settings(root: Path) -> tuple[str, batch_settings, float]:
    model_name = (root / "model_name.dat").read_text().strip()
    exploit, explore, explore_classifier = (
        int(v) for v in np.loadtxt(root / "batch_settings.dat", dtype=int)
    )
    return (
        model_name,
        batch_settings(exploit, explore, explore_classifier),
        float(np.loadtxt(root / "wait_time.dat")),
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
    """``checkComplete.py``: declare jobs that outlived ``wait_time`` complete and infeasible."""
    wait_hours = float(np.loadtxt(root / "wait_time.dat"))
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
        _num(f / "batch_id.dat", 0)
        for f in iter_folders(root, model_name)
        if _num(f / "complete.dat", 0) == 0
    ]
    for batch_id, limit in ((1, batch.exploit), (2, batch.explore), (3, batch.explore_classifier)):
        if busy.count(batch_id) < limit:
            return batch_id
    return 4


def update_db(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``updateDb.py``: gather every job into ``x_obs, y_obs, f_obs, c_obs``; write ``*.dat``."""
    model_name, _, _ = read_settings(root)
    rows = [read_folder(f) for f in iter_folders(root, model_name)]
    x_obs = np.vstack([r[0] for r in rows])
    y_obs, f_obs, c_obs = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    np.savetxt(root / "x_obs.dat", x_obs, fmt="%.16f", delimiter=",")
    np.savetxt(root / "y_obs.dat", y_obs, fmt="%.16f", delimiter=",")
    np.savetxt(root / "f_obs.dat", f_obs, fmt="%d", delimiter=",")
    np.savetxt(root / "c_obs.dat", c_obs, fmt="%d", delimiter=",")
    return x_obs, y_obs, f_obs, c_obs


def wait_for_slot(root: Path, check_time_minutes: float) -> int:
    """Poll until ``get_batch`` returns a free slot (the master loop's ``while batch_id == 4``)."""
    while True:
        check_complete(root)
        batch_id = get_batch(root)
        if batch_id != 4:
            return batch_id
        time.sleep(check_time_minutes * 60)
