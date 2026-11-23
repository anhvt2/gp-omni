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
from gp_omni.asyncbo.acquisition import acquisition_value, check_multiple_design_sites


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(0)
    S = rng.random((12, 2))
    Y = -np.sum((S - 0.5) ** 2, axis=1)
    F = (S[:, 0] < 0.8).astype(float)
    obj = OmniGP().fit(torch.as_tensor(S), torch.as_tensor(Y), max_iter=60)
    clf = OmniGPC().fit(torch.as_tensor(S), torch.as_tensor(F > 0), max_iter=60)
    return S, Y, obj, clf


@pytest.mark.parametrize("acq", ["PI", "EI", "UCB"])
def test_penalty_is_known_times_unknown(models, acq):
    S, Y, obj, clf = models
    x = np.array([0.4, 0.6])
    base = negative_acquisition(x, obj, None, acq, S, Y)
    p = float(clf.predict_proba(torch.as_tensor(x).reshape(1, -1))[0])
    assert 0 < p < 1
    both = negative_acquisition(x, obj, clf, acq, S, Y, known_constraint=lambda _: 1.0)
    assert both == pytest.approx(base * p)
    assert negative_acquisition(x, obj, clf, acq, S, Y, known_constraint=lambda _: 0.0) == 0.0
    # a graded known constraint scales the value linearly
    half = negative_acquisition(x, obj, clf, acq, S, Y, known_constraint=lambda _: 0.5)
    assert half == pytest.approx(0.5 * both)


def test_acquisition_values():
    assert acquisition_value("PI", 1.0, 1.0, 1.0, 5, 2) == pytest.approx(0.5)
    assert acquisition_value("EI", 1.0, 1.0, 1.0, 5, 2) == pytest.approx(0.3989422804)
    assert acquisition_value("EI", 2.0, 0.0, 1.0, 5, 2) == 1.0  # zero variance: no NaN
    with pytest.raises(ValueError):
        acquisition_value("XX", 0.0, 1.0, 0.0, 5, 2)


def test_duplicate_detection():
    S = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert check_multiple_design_sites(np.array([1.0, 1.00001]), S)
    assert not check_multiple_design_sites(np.array([0.5, 0.5]), S)


def make_job(root, i, x, y=None, feasible=1, complete=1, batch=1, scheme=None):
    f = root / f"m_Iter{i}"
    f.mkdir()
    np.savetxt(f / "input.dat", [x], delimiter=",")
    (f / "complete.dat").write_text(f"{complete}\n")
    (f / "batchID.dat").write_text(f"{batch}\n")
    if y is not None:
        (f / "output.dat").write_text(f"{y}\n")
        (f / "feasible.dat").write_text(f"{feasible}\n")
    if scheme:
        (f / "acquisitionScheme.dat").write_text(scheme)


def test_slots_only_count_in_flight_jobs(tmp_path):
    db.write_settings(tmp_path, "m", db.BatchSettings(1, 1, 0), wait_time=1.0)
    assert db.get_batch(tmp_path) == 1
    make_job(tmp_path, 1, [0, 0], complete=0, batch=1)  # running acquisition job
    assert db.get_batch(tmp_path) == 2
    make_job(tmp_path, 2, [1, 1], complete=0, batch=2)
    assert db.get_batch(tmp_path) == 4  # classifier slots = 0, all busy
    (tmp_path / "m_Iter1" / "complete.dat").write_text("1\n")  # one finishes -> slot frees
    assert db.get_batch(tmp_path) == 1


def test_update_db_and_pending_are_zero(tmp_path):
    db.write_settings(tmp_path, "m", db.BatchSettings(1, 0, 0), wait_time=1.0)
    make_job(tmp_path, 10, [3, 3], y=2.0)
    make_job(tmp_path, 2, [1, 2], y=1.0)  # natural sort: Iter2 before Iter10
    make_job(tmp_path, 3, [5, 5], complete=0)  # pending: no output yet
    S, Y, F, C = db.update_db(tmp_path)
    assert S.tolist() == [[1, 2], [5, 5], [3, 3]]
    assert Y.tolist() == [1.0, 0.0, 2.0] and F.tolist() == [1, 0, 1] and C.tolist() == [1, 0, 1]


def test_timeout_flips_pending_job(tmp_path):
    db.write_settings(tmp_path, "m", db.BatchSettings(1, 0, 0), wait_time=1e-6)
    make_job(tmp_path, 1, [0, 0], complete=0)
    (tmp_path / "m_Iter1" / "log.timeout").write_text("")
    time.sleep(0.01)
    db.check_complete(tmp_path)
    assert db.read_folder(tmp_path / "m_Iter1")[1:] == (0.0, 0.0, 1.0)


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
        (tmp_path / f"m_Iter{i}").mkdir()
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

    cfg = AsyncBOConfig(
        model_name="m",
        xLB=[0, 0],
        xUB=[1, 1],
        exploit_size=2,
        explore_size=1,
        explore_clf_size=1,
        max_iter=14,
        num_init_point=4,
        check_time=0.005,
        gp_max_iter=40,
        cma_max_evals=40,
        seed=1,
        known_constraint=lambda x: float(x[0] + x[1] <= 1.5),
    )
    x, y = AsyncBO(tmp_path, cfg, evaluate=simulation).run()
    assert len(calls) == 14
    assert peak[0] > 1  # proposals were made while other jobs were still running
    assert peak[0] <= cfg.batch.total + cfg.num_init_point
    assert x[0] < 0.8 and y > -0.2


def test_requires_exactly_one_launcher(tmp_path):
    cfg = AsyncBOConfig(model_name="m", xLB=[0, 0], xUB=[1, 1])
    with pytest.raises(ValueError):
        AsyncBO(tmp_path, cfg)
