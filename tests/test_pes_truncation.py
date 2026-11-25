"""Independent integration checks for extreme PES truncations."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from gp_omni.opt.acquisition.single_objective import predictive_entropy_search as pes


def tail_excess_moments(tail):
    # For Z >= t, U=t*(Z-t) has density proportional to exp(-u-u^2/(2*t^2)).
    # Integrating U avoids underflow in the tiny probability P(Z >= t).
    moments = [
        quad(lambda u, k=k: u**k * math.exp(-u - u**2 / (2 * tail**2)), 0, np.inf)[0]
        for k in range(3)
    ]
    mean = moments[1] / moments[0]
    return mean / tail, (moments[2] / moments[0] - mean**2) / tail**2


@pytest.mark.parametrize("tail", [8.0, 38.0, 40.0, 100.0, 101.0, 1000.0, 1e6])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_extreme_truncated_moments_match_quadrature(tail, sign):
    bound, sd = 0.7, 1.3
    excess, variance = tail_excess_moments(tail)
    mean = bound - sign * sd * tail
    actual_mean, actual_var = pes.truncated_moments(mean, sd**2, bound, sign)
    assert actual_mean == pytest.approx(bound + sign * sd * excess, abs=1e-10)
    assert actual_var == pytest.approx(sd**2 * variance, rel=1e-7)
    assert sign * (actual_mean - bound) > 0


def test_ep_retains_an_extreme_optimum_constraint():
    expected_mean, expected_var = tail_excess_moments(40.0)
    nu, tau = pes.ep_truncations(
        np.array([-40.0]), np.array([[1.0]]), np.array([0.0]), np.array([1.0])
    )
    assert (nu[0] - 40) / (tau[0] + 1) == pytest.approx(expected_mean, rel=5e-6)
    assert 1 / (tau[0] + 1) == pytest.approx(expected_var, rel=5e-6)


def test_candidate_variance_retains_an_extreme_prediction_constraint():
    m = pes.se_model(
        x=np.empty((0, 1)), r=np.empty(0), mean=0.0, lam=np.ones(1),
        sf2=1.0, sn2=0.0, sd=1.0,
    )
    # At this separation the SE covariance underflows to zero. The candidate and
    # optimum values are independent unit-variance Gaussians, with mean gap -40*sqrt(2).
    state = pes.optimum_state(
        xs=np.array([0.0]), k_inv=np.zeros((1, 1)), beta=np.zeros(1),
        a_q=np.zeros((1, 2)), k_qq=np.eye(2), s_q=np.eye(2),
        m_q=np.array([-40 * math.sqrt(2), 0.0]), nu=np.zeros(2), tau=np.zeros(2),
    )
    _, factor = tail_excess_moments(40.0)
    # Given T=f* - f >= 0, Var(f) = 1/2 + Var(T | T>=0)/4.
    expected = 0.5 + 0.5 * factor
    assert pes.conditional_variance(m, state, np.array([40.0])) == pytest.approx(expected)
