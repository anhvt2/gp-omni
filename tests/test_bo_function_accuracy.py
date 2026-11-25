"""Regression guards on GPR accuracy against analytical benchmark truth.

Thresholds live in ``bo_functions/cases.py`` (about 2x the observed NRMSE, median over seeds).
They guard against regressions; they are not claims that the model is accurate. Run
``python -m bo_functions.report`` from ``tests/`` for the full table including the hard cases.
"""

import math
import statistics

import pytest
from bo_functions import accuracy as acc
from bo_functions import cases as C

import gp_omni  # noqa: F401


def median(values):
    return statistics.median(values)


def _id(case):
    extra = getattr(case, "fn_name", None) or getattr(case, "problem_name", "")
    return f"{extra}-{getattr(case, 'context', '')}".strip("-").replace(" ", "_")


@pytest.mark.parametrize("case", C.SINGLE_CASES, ids=_id)
def test_single_fidelity_accuracy(case):
    fn = case.function()
    kw = dict(case.gpr)
    nrmse = median(
        [acc.single_fidelity_accuracy(fn, case.n_train, seed=s, **kw)["nrmse"] for s in C.SEEDS]
    )
    assert nrmse < case.guard, f"{case.fn_name}: median nrmse {nrmse:.4f} >= guard {case.guard}"
    if case.vs_plain is not None:
        plain = median(
            [acc.single_fidelity_accuracy(fn, case.n_train, seed=s)["nrmse"] for s in C.SEEDS]
        )
        assert nrmse < case.vs_plain * plain, (
            f"{case.fn_name}: additive {nrmse:.4f} should beat plain {plain:.4f} by {case.vs_plain}"
        )


@pytest.mark.parametrize("case", C.MULTI_FIDELITY_CASES, ids=lambda c: f"{c.fn_name}-{c.n_per_level}")
def test_multi_fidelity_accuracy(case):
    runs = [acc.multi_fidelity_accuracy(case.function(), case.n_per_level, seed=s) for s in C.SEEDS]
    mf_nrmse = median([r["mf_nrmse"] for r in runs])
    assert mf_nrmse < case.guard, f"{case.fn_name}: mf nrmse {mf_nrmse:.4f} >= guard {case.guard}"
    if case.min_gain is not None:
        gain = median([r["gain"] for r in runs])
        assert gain > case.min_gain, f"{case.fn_name}: gain {gain:.2f} <= {case.min_gain}"


@pytest.mark.parametrize("case", C.MULTI_OBJECTIVE_CASES, ids=_id)
def test_multi_objective_accuracy(case):
    runs = [acc.multi_objective_accuracy(case.function(), case.n_train, seed=s) for s in C.SEEDS]
    nrmse = median([r["joint_mean_nrmse"] for r in runs])
    assert nrmse < case.guard, f"{case.fn_name}: joint mean nrmse {nrmse:.4f} >= {case.guard}"


@pytest.mark.parametrize("case", C.MULTI_OBJECTIVE_SPARSE_CASES, ids=_id)
def test_multi_objective_with_a_data_poor_objective_stays_finite_and_bounded(case):
    runs = [
        acc.multi_objective_accuracy(
            case.function(), case.n_train, seed=s, sparse_last=case.sparse_last
        )
        for s in C.SEEDS
    ]
    last = median([r["joint_nrmse"][-1] for r in runs])
    assert last < case.guard, f"{case.fn_name}: data-poor objective nrmse {last:.4f}"


@pytest.mark.parametrize("case", C.NOISY_CASES, ids=_id)
def test_noisy_regression_accuracy(case):
    nrmse = median([acc.noisy_accuracy(case.problem(), seed=s)["nrmse"] for s in C.SEEDS])
    assert math.isfinite(nrmse)
    if case.guard is not None:
        assert nrmse < case.guard, f"{case.problem_name}: nrmse {nrmse:.4f} >= {case.guard}"
