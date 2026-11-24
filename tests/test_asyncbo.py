"""Tests for the asynchronous parallel constrained GP-Hedge BO port."""

import threading
import time

import numpy as np
import pytest
import torch

import gp_omni  # noqa: F401
from gp_omni import OmniGP, OmniGPC
from gp_omni.asyncbo import AsyncBO, AsyncBOConfig, negative_acquisition
from gp_omni.asyncbo import database as db
from gp_omni.asyncbo.acquisition import (
    acquisition_value,
    check_multiple_design_sites,
    predict_feasibility,
)


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(0)
    x_obs = rng.random((12, 2))
    y_obs = -np.sum((x_obs - 0.5) ** 2, axis=1)
    f_obs = (x_obs[:, 0] < 0.8).astype(float)
    obj = OmniGP().fit(torch.as_tensor(x_obs), torch.as_tensor(y_obs), max_iter=60)
    classifier = OmniGPC().fit(torch.as_tensor(x_obs), torch.as_tensor(f_obs > 0), max_iter=60)
    return x_obs, y_obs, obj, classifier


@pytest.mark.parametrize("acquisition", ["PI", "EI", "UCB"])
def test_penalty_is_known_times_unknown(models, acquisition):
    x_obs, y_obs, obj, classifier = models
    x = np.array([0.4, 0.6])
    base = negative_acquisition(x, obj, None, acquisition, x_obs, y_obs)
    p = float(classifier.predict_proba(torch.as_tensor(x).reshape(1, -1))[0])
    assert 0 < p < 1
    both = negative_acquisition(
        x, obj, classifier, acquisition, x_obs, y_obs, known_constraint=lambda _: 1.0
    )
    assert both == pytest.approx(base * p)
    assert (
        negative_acquisition(
            x, obj, classifier, acquisition, x_obs, y_obs, known_constraint=lambda _: 0.0
        )
        == 0.0
    )
    # a graded known constraint scales the value linearly
    half = negative_acquisition(
        x, obj, classifier, acquisition, x_obs, y_obs, known_constraint=lambda _: 0.5
    )
    assert half == pytest.approx(0.5 * both)


def test_acquisition_values():
    assert acquisition_value("PI", 1.0, 1.0, 1.0, 5, 2) == pytest.approx(0.5)
    assert acquisition_value("EI", 1.0, 1.0, 1.0, 5, 2) == pytest.approx(0.3989422804)
    assert acquisition_value("EI", 2.0, 0.0, 1.0, 5, 2) == 1.0  # zero variance: no NaN
    with pytest.raises(ValueError):
        acquisition_value("XX", 0.0, 1.0, 0.0, 5, 2)


def test_duplicate_detection():
    x_obs = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert check_multiple_design_sites(np.array([1.0, 1.00001]), x_obs)
    assert not check_multiple_design_sites(np.array([0.5, 0.5]), x_obs)


def make_job(root, i, x, y=None, feasible=1, complete=1, batch=1, scheme=None):
    f = root / f"m_iter{i}"
    f.mkdir()
    np.savetxt(f / "input.dat", [x], delimiter=",")
    (f / "complete.dat").write_text(f"{complete}\n")
    (f / "batch_id.dat").write_text(f"{batch}\n")
    if y is not None:
        (f / "output.dat").write_text(f"{y}\n")
        (f / "feasible.dat").write_text(f"{feasible}\n")
    if scheme:
        (f / "acquisition_scheme.dat").write_text(scheme)


def test_slots_only_count_in_flight_jobs(tmp_path):
    db.write_settings(tmp_path, "m", db.BatchSettings(1, 1, 0), wait_time=1.0)
    assert db.get_batch(tmp_path) == 1
    make_job(tmp_path, 1, [0, 0], complete=0, batch=1)  # running acquisition job
    assert db.get_batch(tmp_path) == 2
    make_job(tmp_path, 2, [1, 1], complete=0, batch=2)
    assert db.get_batch(tmp_path) == 4  # classifier slots = 0, all busy
    (tmp_path / "m_iter1" / "complete.dat").write_text("1\n")  # one finishes -> slot frees
    assert db.get_batch(tmp_path) == 1


def test_update_db_and_pending_are_zero(tmp_path):
    db.write_settings(tmp_path, "m", db.BatchSettings(1, 0, 0), wait_time=1.0)
    make_job(tmp_path, 10, [3, 3], y=2.0)
    make_job(tmp_path, 2, [1, 2], y=1.0)  # natural sort: Iter2 before Iter10
    make_job(tmp_path, 3, [5, 5], complete=0)  # pending: no output yet
    x_obs, y_obs, f_obs, c_obs = db.update_db(tmp_path)
    assert x_obs.tolist() == [[1, 2], [5, 5], [3, 3]]
    assert (
        y_obs.tolist() == [1.0, 0.0, 2.0]
        and f_obs.tolist() == [1, 0, 1]
        and c_obs.tolist() == [1, 0, 1]
    )


def test_timeout_flips_pending_job(tmp_path):
    db.write_settings(tmp_path, "m", db.BatchSettings(1, 0, 0), wait_time=1e-6)
    make_job(tmp_path, 1, [0, 0], complete=0)
    (tmp_path / "m_iter1" / "log.timeout").write_text("")
    time.sleep(0.01)
    db.check_complete(tmp_path)
    assert db.read_folder(tmp_path / "m_iter1")[1:] == (0.0, 0.0, 1.0)


def test_rewards_credit_new_bests_only(tmp_path):
    db.write_settings(tmp_path, "m", db.BatchSettings(1, 0, 0), wait_time=1.0)
    make_job(tmp_path, 1, [0, 0], y=1.0, batch=0)
    make_job(tmp_path, 2, [0, 1], y=2.0, scheme="EI")  # new best -> EI
    make_job(tmp_path, 3, [0, 2], y=1.5, scheme="UCB")  # not a new best
    make_job(tmp_path, 4, [0, 3], y=0, feasible=0, scheme="PI")  # infeasible
    make_job(tmp_path, 5, [0, 4], y=3.0, scheme="EI")  # new best -> EI
    make_job(tmp_path, 6, [0, 5], complete=0, scheme="UCB")  # no output yet
    assert db.get_rewards(tmp_path, num_init_point=1).tolist() == [0.0, 2.0, 0.0]


def test_hedge_scheme_follows_rewards(tmp_path):
    db.write_settings(tmp_path, "m", db.BatchSettings(1, 0, 0), wait_time=1.0)
    for i in range(1, 400):
        (tmp_path / f"m_iter{i}").mkdir()
    rng = np.random.default_rng(0)
    picks = [
        db.get_acquisition_scheme(tmp_path, 1, np.array([0.0, 50.0, 0.0]), rng) for _ in range(200)
    ]
    assert picks.count("EI") > 150


def test_asynchronous_overlap_and_penalised_search(tmp_path):
    lock, running, peak, calls = threading.Lock(), [0], [0], []

    def simulation(x):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.15)
        with lock:
            running[0] -= 1
            calls.append(x.copy())
        return -np.sum((x - 0.3) ** 2), bool(x[0] < 0.8)  # unknown constraint x0 < 0.8

    config = AsyncBOConfig(
        model_name="m",
        x_lb=[0, 0],
        x_ub=[1, 1],
        exploit_size=2,
        explore_size=1,
        explore_classifier_size=1,
        max_iter=14,
        num_init_point=4,
        check_time=0.005,
        gp_max_iter=40,
        cma_max_evals=40,
        seed=1,
        known_constraint=lambda x: float(x[0] + x[1] <= 1.5),
    )
    x, y = AsyncBO(tmp_path, config, evaluate=simulation).run()
    assert len(calls) == 14
    assert peak[0] > 1  # proposals were made while other jobs were still running
    assert peak[0] <= config.batch.total + config.num_init_point
    assert x[0] < 0.8 and y > -0.2


def test_requires_exactly_one_launcher(tmp_path):
    config = AsyncBOConfig(model_name="m", x_lb=[0, 0], x_ub=[1, 1])
    with pytest.raises(ValueError):
        AsyncBO(tmp_path, config)


def test_end_to_end_with_shell_script_jobs(tmp_path):
    """Full file protocol: template copied per job, ``query.sh`` detaches a background job."""
    import stat
    import subprocess
    import sys

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

    config = AsyncBOConfig(
        model_name="toy", x_lb=[0, 0], x_ub=[1, 1], exploit_size=2, explore_size=1,
        explore_classifier_size=1, max_iter=12, num_init_point=3, check_time=0.005,
        gp_max_iter=40, cma_max_evals=40, seed=2, query_script="query.sh",
    )  # fmt: skip
    x, y = AsyncBO(tmp_path, config).run()

    jobs = db.iter_folders(tmp_path, "toy")
    assert len(jobs) == 12
    assert all(db.read_folder(job)[3] == 1 for job in jobs)  # every job finished
    assert not subprocess.run(["pgrep", "-f", "simulate.py"], capture_output=True).stdout
    for name in ("x_obs", "y_obs", "f_obs", "c_obs"):
        assert (tmp_path / f"{name}.dat").is_file()
    proposals = jobs[3:]
    assert all((job / "gp_predictions.dat").is_file() for job in proposals)
    hedge = [job for job in proposals if (job / "batch_id.dat").read_text().strip() == "1"]
    assert hedge and all(
        (job / "acquisition_scheme.dat").read_text() in ("UCB", "EI", "PI") for job in hedge
    )
    assert x[0] < 0.8 and y > -0.3


def test_config_json_round_trip(tmp_path):
    from pathlib import Path

    shipped = Path(__file__).parents[1] / "config" / "async_bo.json"
    config = AsyncBOConfig.from_json(shipped, known_constraint=lambda _: 1.0)
    assert config.model_name == "rosen" and config.x_lb.tolist() == [-1.0, -1.0]
    assert config.x_scale.tolist() == [1.0, 1.0] and config.batch.total == 4
    config.to_json(tmp_path / "copy.json")
    assert AsyncBOConfig.from_json(tmp_path / "copy.json").x_ub.tolist() == config.x_ub.tolist()


def test_relaxed_feasibility_probability():
    class Fixed:
        def __init__(self, p):
            self.p = p

        def predict_proba(self, xs):
            return torch.tensor([self.p])

    x = np.zeros(2)
    assert predict_feasibility(Fixed(0.95), x, relax_alpha=0.9) == 1.0
    assert predict_feasibility(Fixed(0.05), x, relax_alpha=0.9) == 0.0
    assert predict_feasibility(Fixed(0.5), x, relax_alpha=0.9) == 0.5
    assert predict_feasibility(Fixed(0.95), x) == 0.95  # off by default


def test_relax_alpha_reaches_the_penalty(models):
    x_obs, y_obs, objective, _ = models

    class Confident:  # classifier that is 95% sure the point is feasible
        def predict_proba(self, xs):
            return torch.tensor([0.95])

    x = np.array([0.4, 0.6])
    plain = negative_acquisition(x, objective, None, "EI", x_obs, y_obs)
    soft = negative_acquisition(x, objective, Confident(), "EI", x_obs, y_obs)
    rounded = negative_acquisition(x, objective, Confident(), "EI", x_obs, y_obs, relax_alpha=0.9)
    assert soft == pytest.approx(0.95 * plain)
    assert rounded == pytest.approx(plain)  # 0.95 > 0.9 is rounded up to 1
