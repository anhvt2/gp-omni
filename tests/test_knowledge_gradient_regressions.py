"""Independent checks of KG's decision set and observation noise."""

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import norm

from gp_omni.opt.acquisition.single_objective import kg


class GaussianBelief:
    def __init__(self, means, covariance, noise):
        self.means = torch.as_tensor(means, dtype=torch.float64)
        self.covariance = torch.as_tensor(covariance, dtype=torch.float64)
        self.noise = noise

    def predict(self, xs, full_cov=False):
        assert full_cov and len(xs) == len(self.means)
        variance = self.covariance.diagonal()
        return {
            "fmu": self.means,
            "fcov": self.covariance,
            "fs2": variance,
            "ys2": variance + self.noise,
        }


@pytest.mark.parametrize("query_mean", [-2.0, 0.0, 2.0])
def test_independent_query_cannot_improve_a_single_fixed_decision(query_mean):
    belief = GaussianBelief([0.0, query_mean], np.eye(2), noise=0.0)
    assert kg.knowledge_gradient(np.array([1.0]), belief, np.array([[0.0]])) == 0.0


@pytest.mark.parametrize("noise", [0.0, 0.2, 4.0])
def test_fixed_set_kg_matches_numerical_gaussian_integration(noise):
    means = np.array([0.2, -0.1, 2.0])
    covariance = np.array([[1.0, 0.0, 0.3], [0.0, 1.0, -0.5], [0.3, -0.5, 1.0]])
    belief = GaussianBelief(means, covariance, noise)
    slopes = covariance[:-1, -1] / np.sqrt(1 + noise)
    baseline = means[:-1].max()
    reference = quad(
        lambda z: (np.max(means[:-1] + slopes * z) - baseline) * norm.pdf(z),
        -12,
        12,
        epsabs=1e-10,
    )[0]
    actual = kg.knowledge_gradient(np.array([0.5]), belief, np.array([[0.0], [1.0]]))
    assert actual == pytest.approx(reference, abs=1e-9)
