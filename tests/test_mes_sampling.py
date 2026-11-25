"""Check MES's Gumbel calibration against known maximum-distribution quantiles."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from gp_omni.opt.acquisition.single_objective.pointwise import sample_max_values


class StandardNormalMarginals:
    def predict(self, x):
        return {"ymu": torch.zeros(len(x)), "ys2": torch.ones(len(x))}


class QuantileRNG:
    def __init__(self, probability):
        self.probability = probability

    def random(self, size):
        return np.full(size, self.probability)


@pytest.mark.parametrize("n_candidates", [10, 1000])
def test_sampled_median_matches_the_maximum_distribution(n_candidates):
    # P(max <= y) = Phi(y)^n, so the median is available independently of the Gumbel fit.
    expected = norm.ppf(0.5 ** (1 / n_candidates))
    actual = sample_max_values(
        StandardNormalMarginals(), [0.0], [1.0], -100.0, QuantileRNG(0.5),
        n_samples=1, n_candidates=n_candidates,
    )
    assert actual[0] == pytest.approx(expected, abs=1e-9)


def test_sampled_interquartile_range_matches_the_maximum_distribution():
    samples = [
        sample_max_values(
            StandardNormalMarginals(), [0.0], [1.0], -100.0, QuantileRNG(p), n_samples=1,
        )[0]
        for p in (0.25, 0.75)
    ]
    expected = norm.ppf(0.75 ** (1 / 1000)) - norm.ppf(0.25 ** (1 / 1000))
    assert samples[1] - samples[0] == pytest.approx(expected, abs=1e-9)


def test_calibrated_samples_still_respect_the_incumbent_floor():
    actual = sample_max_values(
        StandardNormalMarginals(), [0.0], [1.0], 5.0, QuantileRNG(0.5), n_samples=1,
    )
    assert actual[0] == 5.0
