"""The structured task covariance: multi-output and multi-fidelity as independent switches."""

import math

import numpy as np
import pytest
import torch

import gp_omni  # noqa: F401
from gp_omni.build import ar1_cokriging
from gp_omni.build.covariance import covariance
from gp_omni.build.hyperparameters import make_hyperparameters
from gp_omni.build.mean import mean_function
from gp_omni.build.multitask import (
    augment,
    task_covariance,
    task_expand,
    task_gaussian,
    task_mean,
)
from gp_omni.build.sparse_covariance import sparse_covariance
from gp_omni.infer.moments import predict_gp
from gp_omni.infer.posterior import gaussian_posterior
from gp_omni.train.objectives import negative_log_evidence


def branin_pair(n_lo=14, n_hi=6, seed=0):
    rng = np.random.default_rng(seed)

    def hi(x):
        x1, x2 = 15 * x[:, 0] - 5, 15 * x[:, 1]
        return (
            (x2 - 5.1 / (4 * np.pi**2) * x1**2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1)
            + 10
        )

    def lo(x):
        return 0.8 * hi(x) + 5 * (x[:, 0] - 0.5) - 3

    xl, xh = rng.random((n_lo, 2)), rng.random((n_hi, 2))
    return [torch.tensor(xl), torch.tensor(xh)], [torch.tensor(lo(xl)), torch.tensor(hi(xh))]


def test_single_task_reduces_to_base_covariance():
    g = torch.Generator().manual_seed(0)
    x, z = torch.rand(7, 3, generator=g), torch.rand(5, 3, generator=g)
    base = covariance("matern_ard", 5)
    h = torch.tensor(np.log([0.5, 0.7, 0.9, 1.3]))
    tc = task_covariance(base, 3)
    torch.testing.assert_close(tc(h, augment(x), augment(z)), base(h, x, z))
    torch.testing.assert_close(tc(h, augment(x), "diag"), base(h, x, "diag"))


def test_two_fidelity_equals_ar1_cokriging_at_fixed_hyperparameters():
    X, Y = branin_pair()
    theta1, theta2, rho = np.array([0.3, 0.1]), np.array([0.6, 0.2]), 1.4
    ar1 = ar1_cokriging(
        X,
        Y,
        correlation="matern32",
        regression="none",
        hyperparameters={"theta1": theta1, "theta2": theta2, "rho": rho},
    ).fit()
    # MATLAB precision 10**hp <-> gpml length scale ell = 10**(-hp/2); sf^2 = process variance
    log_ell = lambda th: -th * math.log(10) / 2
    sf = lambda s2: 0.5 * math.log(float(s2))
    tc = task_covariance(covariance("matern_ard", 3), 2, num_fidelities=2)
    hyp = torch.cat(
        [
            torch.tensor(log_ell(theta1)),
            torch.tensor([sf(ar1.level1.sigma2)]),
            torch.tensor(log_ell(theta2)),
            torch.tensor([sf(ar1.level2.sigma2)]),
            torch.tensor([rho]),
        ]
    )
    assert hyp.numel() == tc.num_hyp()
    xa = torch.cat([augment(X[0], None, torch.zeros(14)), augment(X[1], None, torch.ones(6))])
    y = torch.cat(Y)
    h = make_hyperparameters(mean=np.zeros(2), cov=hyp, lik=np.full(2, -8.0))
    lik, mean = task_gaussian(2, 1, 2), task_mean(2, 1, 2)
    _, _, post = negative_log_evidence(h, gaussian_posterior, mean, tc, lik, xa, y, grad=False)
    xs = torch.tensor(np.random.default_rng(1).random((12, 2)))
    p = predict_gp(h, post, mean, tc, lik, xa, augment(xs, None, torch.ones(12)))
    m_ref, v_ref = ar1.predict(xs)
    scale = float(m_ref.abs().max()) + 1
    np.testing.assert_allclose(p["fmu"].numpy(), m_ref.numpy(), atol=1e-5 * scale)
    np.testing.assert_allclose(p["fs2"].numpy(), v_ref.numpy(), rtol=1e-3, atol=1e-4 * scale**2)


@pytest.mark.parametrize("O,L", [(1, 1), (2, 1), (1, 3), (3, 2)])
def test_covariance_is_symmetric_psd_and_diag_consistent(O, L):
    g = torch.Generator().manual_seed(1)
    D, n = 2, 18
    tc = task_covariance(covariance("se_ard"), D, O, L, rank=2)
    h = tc.initial_hyp(torch.tensor(np.log([0.6, 0.8, 1.0])))
    h[-O * (L - 1) :] = torch.linspace(0.5, 1.5, O * (L - 1)) if L > 1 else h[-0:]
    x = augment(
        torch.rand(n, D, generator=g),
        torch.randint(0, O, (n,), generator=g),
        torch.randint(0, L, (n,), generator=g),
    )
    K = tc(h, x)
    torch.testing.assert_close(K, K.T)
    assert torch.linalg.eigvalsh(K).min() > -1e-9
    torch.testing.assert_close(tc(h, x, "diag"), torch.diagonal(K))


def test_fidelities_are_nested_through_rho():
    """f_hi = rho f_lo + g_1: cov(hi, lo) = rho * cov(lo, lo) and rho = 0 decouples."""
    g = torch.Generator().manual_seed(2)
    tc = task_covariance(covariance("se_iso"), 2, num_fidelities=2)
    base = torch.tensor([math.log(0.5), 0.0])
    x = torch.rand(6, 2, generator=g)
    lo, hi = augment(x, None, torch.zeros(6)), augment(x, None, torch.ones(6))
    for rho in (1.7, 0.0):
        h = torch.cat([base, base, torch.tensor([rho])])
        torch.testing.assert_close(tc(h, hi, lo), rho * tc(h, lo, lo))
    assert torch.allclose(tc(h, hi, lo), torch.zeros(6, 6))


def test_sparse_with_all_locations_matches_exact_multitask():
    g = torch.Generator().manual_seed(3)
    D, O, L, n = 2, 2, 2, 40
    x = torch.rand(n, D, generator=g)
    o, l = torch.randint(0, O, (n,), generator=g), torch.randint(0, L, (n,), generator=g)
    y = torch.sin(4 * x[:, 0]) + 0.3 * o + 0.5 * l + 0.05 * torch.randn(n, generator=g)
    xa = augment(x, o, l)
    tc = task_covariance(covariance("se_ard"), D, O, L)
    h0 = tc.initial_hyp(torch.tensor(np.log([0.6, 0.8, 1.0])))
    base = {"mean": np.zeros(O * L), "cov": h0, "lik": np.full(O * L, math.log(0.1))}
    mean, lik = task_mean(D, O, L), task_gaussian(D, O, L)
    exact, _, post_e = negative_log_evidence(
        make_hyperparameters(**base), gaussian_posterior, mean, tc, lik, xa, y, grad=False
    )
    sc = sparse_covariance(tc, x.clone(), expand=task_expand(D, O, L))  # every location, every task
    sparse, _, post_s = negative_log_evidence(
        make_hyperparameters(**base),
        gaussian_posterior,
        mean,
        sc,
        lik,
        xa,
        y,
        {"s": 0.0},
        grad=False,
    )
    assert float(sparse) == pytest.approx(float(exact), abs=1e-3)
    xs = augment(torch.rand(9, D, generator=g), torch.ones(9), torch.ones(9))
    pe = predict_gp(make_hyperparameters(**base), post_e, mean, tc, lik, xa, xs)
    ps = predict_gp(make_hyperparameters(**base), post_s, mean, sc, lik, xa, xs)
    np.testing.assert_allclose(ps["fmu"].numpy(), pe["fmu"].numpy(), atol=1e-4)


def test_per_task_noise_is_used():
    x = augment(torch.rand(4, 2), torch.tensor([0, 1, 0, 1]), torch.zeros(4))
    lik = task_gaussian(2, 2, 1)
    nv = lik.noise_var(torch.tensor([math.log(0.1), math.log(0.5)]), x)
    np.testing.assert_allclose(nv.numpy(), [0.01, 0.25, 0.01, 0.25])
    assert mean_function("const").num_hyp(2) == 1 and task_mean(2, 2, 1).num_hyp() == 2
