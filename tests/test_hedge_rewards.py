"""Reward accounting must use valid completed jobs, including zero-valued results."""

import numpy as np
import pytest

from gp_omni.opt.config import batch_settings
from gp_omni.opt.parallel import jobs
from gp_omni.opt.parallel.asyncbo import hedge


def make_job(root, i, value, scheme=None, feasible=1, complete=1):
    folder = root / f"m_iter{i}"
    folder.mkdir()
    for filename, number in (
        ("output.dat", value), ("feasible.dat", feasible), ("complete.dat", complete),
    ):
        if number is not None:
            (folder / filename).write_text(f"{number}\n")
    if scheme:
        (folder / "acquisition_scheme.dat").write_text(scheme)


@pytest.mark.parametrize(
    "initial,proposal,expected",
    [([1.0, 10.0], 5.0, 0.0), ([-2.0, -1.0], 0.0, 1.0)],
)
def test_rewards_use_all_initial_values_and_accept_zero(tmp_path, initial, proposal, expected):
    jobs.write_settings(tmp_path, "m", batch_settings(1, 0, 0), 1.0)
    for i, value in enumerate(initial, 1):
        make_job(tmp_path, i, value)
    make_job(tmp_path, 3, proposal, "EI")
    assert hedge.get_rewards(tmp_path, 2, ("EI",)).tolist() == [expected]


def test_infeasible_initial_zero_does_not_suppress_negative_improvements(tmp_path):
    jobs.write_settings(tmp_path, "m", batch_settings(1, 0, 0), 1.0)
    make_job(tmp_path, 1, 0.0, feasible=0)
    make_job(tmp_path, 2, -10.0)
    make_job(tmp_path, 3, -8.0, "EI")
    assert hedge.get_rewards(tmp_path, 2, ("EI",)).tolist() == [1.0]


@pytest.mark.parametrize(
    "invalid",
    [
        {"value": 100.0, "feasible": 0},
        {"value": 100.0, "complete": 0},
        {"value": 100.0, "feasible": None},
        {"value": 100.0, "complete": None},
        {"value": None},
        {"value": np.nan},
        {"value": np.inf},
    ],
)
def test_invalid_jobs_neither_earn_rewards_nor_change_the_incumbent(tmp_path, invalid):
    jobs.write_settings(tmp_path, "m", batch_settings(1, 0, 0), 1.0)
    make_job(tmp_path, 1, 1.0)
    make_job(tmp_path, 2, scheme="EI", **invalid)
    make_job(tmp_path, 3, 2.0, "PES")
    assert hedge.get_rewards(tmp_path, 1, ("EI", "PES")).tolist() == [0.0, 1.0]


def test_no_valid_initial_observations_still_allows_a_first_incumbent(tmp_path):
    jobs.write_settings(tmp_path, "m", batch_settings(1, 0, 0), 1.0)
    make_job(tmp_path, 1, None, feasible=0)
    make_job(tmp_path, 2, -2.0, "EI")
    assert hedge.get_rewards(tmp_path, 1, ("EI",)).tolist() == [1.0]


def test_exploration_results_update_the_incumbent_without_crediting_an_arm(tmp_path):
    jobs.write_settings(tmp_path, "m", batch_settings(1, 0, 0), 1.0)
    make_job(tmp_path, 1, 1.0)
    make_job(tmp_path, 2, 4.0)
    make_job(tmp_path, 3, 3.0, "EI")
    assert hedge.get_rewards(tmp_path, 1, ("EI",)).tolist() == [0.0]


def test_empty_job_pool_has_zero_rewards(tmp_path):
    jobs.write_settings(tmp_path, "m", batch_settings(1, 0, 0), 1.0)
    assert hedge.get_rewards(tmp_path, 3, ("EI", "KG")).tolist() == [0.0, 0.0]
