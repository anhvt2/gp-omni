"""Tests for the batch-sequential parallel constrained BO port."""

import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import gp_omni  # noqa: F401
from gp_omni.asyncbo import database as db
from gp_omni.batchbo import BatchSeqBO, BatchSeqBOConfig


def make_config(**overrides):
    values = {
        "model_name": "toy",
        "x_lb": [0, 0],
        "x_ub": [1, 1],
        "exploit_size": 2,
        "explore_size": 1,
        "explore_classifier_size": 1,
        "max_iter": 6,
        "num_init_point": 3,
        "check_time": 0.005,
        "gp_max_iter": 40,
        "cma_max_evals": 40,
        "seed": 1,
        "known_constraint": lambda x: float(x[0] + x[1] <= 1.5),
    }
    return BatchSeqBOConfig(**{**values, **overrides})


def test_batches_run_in_parallel_but_never_overlap(tmp_path):
    lock, events = threading.Lock(), []

    def simulation(x):
        start = time.monotonic()
        time.sleep(0.15)
        with lock:
            events.append((start, time.monotonic()))
        return -np.sum((x - 0.3) ** 2), bool(x[0] < 0.8)  # unknown constraint x0 < 0.8

    config = make_config()
    x, y = BatchSeqBO(tmp_path, config, evaluate=simulation).run()

    size = config.batch.total
    assert len(events) == config.num_init_point + (config.max_iter - config.num_init_point) * size
    # sorted by finish time, jobs of one batch are consecutive; the next batch starts only
    # after the previous one's last job has finished (the barrier)
    events_by_finish = sorted(events, key=lambda e: e[1])
    ordered = events_by_finish[config.num_init_point :]
    for k in range(2):
        previous, following = ordered[k * size : (k + 1) * size], ordered[(k + 1) * size :][:size]
        assert max(e[1] for e in previous) <= min(e[0] for e in following) + 1e-3
    # within a batch the jobs overlap (parallel)
    first = ordered[:size]
    assert min(e[1] for e in first) > max(e[0] for e in first)
    assert x[0] < 0.8 and y > -0.3


def test_batch_points_are_distinct_thanks_to_hallucination(tmp_path):
    config = make_config(exploit_size=4, explore_size=0, explore_classifier_size=0, max_iter=4)
    BatchSeqBO(tmp_path, config, evaluate=lambda x: (-np.sum((x - 0.3) ** 2), True)).run()
    x_obs, _, _, _ = db.update_db(tmp_path)
    batch = x_obs[config.num_init_point :]
    spread = np.linalg.norm(batch[:, None] - batch[None], axis=-1)[np.triu_indices(4, 1)]
    assert spread.min() > 1e-3
    folders = [p.name for p in db.iter_folders(tmp_path, "toy")]
    assert "toy_iter4_exploit_batch4" in folders


def test_end_to_end_with_shell_script_jobs(tmp_path):
    template = tmp_path / "toy_template"
    template.mkdir()
    (template / "simulate.py").write_text(
        "import time, numpy as np\n"
        "time.sleep(0.2)\n"
        "x = np.loadtxt('input.dat', delimiter=',')\n"
        "open('output.dat', 'w').write(f'{-np.sum((x - 0.3) ** 2):.8f}\\n')\n"
        "open('feasible.dat', 'w').write(f'{int(x[0] < 0.8)}\\n')\n"
        "open('complete.dat', 'w').write('1\\n')\n"
    )
    query = template / "query.sh"
    query.write_text(f"#!/bin/bash\nnohup {sys.executable} simulate.py > log.simulate 2>&1 &\n")
    query.chmod(query.stat().st_mode | stat.S_IXUSR)

    config = make_config(query_script="query.sh", max_iter=5, seed=2)
    x, y = BatchSeqBO(tmp_path, config).run()

    jobs = db.iter_folders(tmp_path, "toy")
    assert len(jobs) == 3 + 2 * 4
    assert all(db.read_folder(job)[3] == 1 for job in jobs)
    assert {p.name for p in jobs if "_iter" in p.name and "_batch" in p.name} >= {
        "toy_iter4_exploit_batch1", "toy_iter5_explore_batch1",
        "toy_iter5_explore_classifier_batch1",
    }  # fmt: skip
    assert x[0] < 0.8 and y > -0.35


def test_shipped_config_loads():
    path = Path(__file__).parents[1] / "config" / "batch_seq_bo.json"
    config = BatchSeqBOConfig.from_json(path)
    assert config.acquisition == "EI" and config.relax_alpha == 0.9 and config.batch.total == 5


def test_launcher_is_required():
    with pytest.raises(ValueError):
        BatchSeqBO(Path("/tmp/unused-batchbo"), make_config())
