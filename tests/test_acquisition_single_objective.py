"""Single-objective acquisition functions in :mod:`gp_omni.opt.acquisition`."""

import math

import numpy as np
import pytest
import torch
from scipy.stats import norm

import gp_omni  # noqa: F401
from gp_omni import GPR
from gp_omni.opt.acquisition import acquisition_value, negative_acquisition, prepare_context
from gp_omni.opt.acquisition import single_objective as so


def test_xi_shifts_pi_and_ei():
    assert acquisition_value("PI", 1.0, 1.0, 0.0, 5, 2, xi=1.0) == pytest.approx(0.5)
    assert acquisition_value("EI", 1.0, 1.0, 0.0, 5, 2, xi=1.0) == pytest.approx(0.3989422804)
    assert acquisition_value("PI", 2.0, 0.0, 1.0, 5, 2) == 1.0
    assert acquisition_value("PI", 1.0, 0.0, 1.0, 5, 2) == 0.0


def test_pe_is_variance_and_ucb_uses_kappa():
    assert acquisition_value("PE", 3.0, 0.7, 0.0, 5, 2) == pytest.approx(0.7)
    kappa = so.ucb_kappa(5, 2)
    assert acquisition_value("UCB", 1.0, 4.0, 0.0, 5, 2) == pytest.approx(1 + 2 * math.sqrt(kappa))


def test_mes_closed_form_and_monotone_in_uncertainty():
    y_star = np.array([1.0, 2.0])
    gamma = (y_star - 0.0) / 1.0
    expected = np.mean(gamma * norm.pdf(gamma) / (2 * norm.cdf(gamma)) - norm.logcdf(gamma))
    assert so.max_value_entropy(0.0, 1.0, y_star) == pytest.approx(expected)
    assert so.max_value_entropy(0.0, 0.0, y_star) == 0.0
    assert so.max_value_entropy(0.0, 4.0, y_star) > so.max_value_entropy(0.0, 0.25, y_star) > 0


def test_mes_requires_context():
    with pytest.raises(ValueError, match="context"):
        acquisition_value("MES", 0.0, 1.0, 0.0, 5, 2)


def test_unknown_name_lists_choices():
    with pytest.raises(ValueError, match="MES"):
        acquisition_value("nope", 0.0, 1.0, 0.0, 5, 2)


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(1)
    x = rng.random((10, 2))
    y = -np.sum((x - 0.5) ** 2, axis=1)
    return x, y, GPR().fit(torch.as_tensor(x), torch.as_tensor(y), max_iter=60)


def test_max_value_samples_are_floored_at_best(model):
    _x, y, gp = model
    rng = np.random.default_rng(0)
    y_star = so.sample_max_values(gp, [0, 0], [1, 1], float(y.max()), rng, n_samples=20)
    assert y_star.shape == (20,)
    assert np.all(y_star >= y.max())
    assert np.all(np.isfinite(y_star))


@pytest.mark.parametrize("name", so.NAMES)
def test_every_acquisition_scores_through_the_penalised_objective(model, name):
    x, y, gp = model
    ctx = prepare_context(name, gp, [0, 0], [1, 1], x, y, np.random.default_rng(0))
    value = negative_acquisition(np.array([0.3, 0.7]), gp, None, name, x, y, context=ctx)
    assert np.isfinite(value)
    if name != "UCB":  # UCB follows the (possibly negative) posterior mean
        assert value <= 0
