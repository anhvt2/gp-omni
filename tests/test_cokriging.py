from pathlib import Path

import numpy as np
import pytest
import torch

import gp_omni  # noqa: F401  (sets float64)
from gp_omni.build import ar1_cokriging
from gp_omni.build.correlation import correlation


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

    xl, xh = rng.random((n_lo, 2)), rng.random((n_hi, 2))  # non-nested
    return [torch.tensor(xl), torch.tensor(xh)], [
        torch.tensor(lo(xl)),
        torch.tensor(hi(xh)),
    ]


def np_corr(name, th, a, b):
    return correlation(name, torch.tensor(th), torch.tensor(a), torch.tensor(b)).numpy()


def np_basis(kind, x):
    n = x.shape[0]
    if kind == "none":
        return np.zeros((n, 0))
    if kind == "regpoly0":
        return np.ones((n, 1))
    return np.hstack([np.ones((n, 1)), x])  # regpoly1


def reference_predict(m, xs, name, kind):
    """Dense GLS co-kriging with explicit inverses, independent of the model code."""
    X1, X2 = (x.numpy() for x in m.X)
    y = np.concatenate([t.numpy() for t in m.Y])
    th1, th2 = m.level1.theta.numpy(), m.level2.theta.numpy()
    s1, s2, rho = float(m.level1.sigma2), float(m.level2.sigma2), m.rho
    S = np.block(
        [
            [s1 * np_corr(name, th1, X1, X1), rho * s1 * np_corr(name, th1, X1, X2)],
            [
                rho * s1 * np_corr(name, th1, X2, X1),
                rho**2 * s1 * np_corr(name, th1, X2, X2) + s2 * np_corr(name, th2, X2, X2),
            ],
        ]
    )
    F1, F2 = np_basis(kind, X1), np_basis(kind, X2)
    F = np.block([[F1, np.zeros_like(F1)], [rho * F2, F2]])
    Si = np.linalg.inv(S)
    if F.shape[1]:
        beta = np.linalg.solve(F.T @ Si @ F, F.T @ Si @ y)
    else:
        beta = np.zeros(0)
    Fx = np_basis(kind, xs)
    Fx = np.hstack([rho * Fx, Fx])
    psi = np.hstack(
        [
            rho * s1 * np_corr(name, th1, xs, X1),
            rho**2 * s1 * np_corr(name, th1, xs, X2) + s2 * np_corr(name, th2, xs, X2),
        ]
    )
    mean = Fx @ beta + psi @ Si @ (y - F @ beta)
    var = rho**2 * s1 + s2 - np.einsum("ij,jk,ik->i", psi, Si, psi)
    if F.shape[1]:
        U = Fx - psi @ Si @ F
        var = var + np.einsum("ij,jk,ik->i", U, np.linalg.inv(F.T @ Si @ F), U)
    return mean, var


FIXED = {"theta1": [0.3, 0.1], "theta2": [0.6, 0.2], "rho": 1.4}


@pytest.mark.parametrize("name", ["gauss", "matern32", "matern52", "exp"])
@pytest.mark.parametrize("kind", ["none", "regpoly0", "regpoly1"])
def test_matches_dense_reference(name, kind):
    X, Y = branin_pair()
    m = ar1_cokriging(X, Y, correlation=name, regression=kind, hyperparameters=FIXED).fit()
    xs = np.random.default_rng(1).random((15, 2))
    mean, var = m.predict(torch.tensor(xs))
    rmean, rvar = reference_predict(m, xs, name, kind)
    scale = np.abs(rmean).max() + 1
    np.testing.assert_allclose(mean.numpy(), rmean, atol=2e-6 * scale)
    np.testing.assert_allclose(var.numpy(), np.clip(rvar, 0, None), rtol=1e-4, atol=1e-5 * scale**2)


def test_fit_interpolates_high_fidelity_and_variance_nonnegative():
    X, Y = branin_pair()
    m = ar1_cokriging(X, Y, correlation="matern32", regression="regpoly0").fit()
    mean, var = m.predict(X[1])
    np.testing.assert_allclose(mean.numpy(), Y[1].numpy(), atol=1e-5 * np.abs(Y[1].numpy()).max())
    assert var.max() < 1e-6 * m.sigma2
    _, v = m.predict(torch.rand(50, 2))
    assert (v >= 0).all()
    ml, _ = m.predict_low(X[0])
    np.testing.assert_allclose(ml.numpy(), Y[0].numpy(), atol=1e-5 * np.abs(Y[0].numpy()).max())


def test_rho_zero_decouples_low_fidelity():
    X, Y = branin_pair()
    hp = dict(FIXED, rho=0.0)
    m = ar1_cokriging(X, Y, correlation="matern32", regression="regpoly0", hyperparameters=hp).fit()
    # Perturbing the low-fidelity outputs must not change high-fidelity predictions.
    Y2 = [Y[0] + 100.0 * torch.randn(Y[0].shape[0]), Y[1]]
    m2 = ar1_cokriging(
        X, Y2, correlation="matern32", regression="regpoly0", hyperparameters=hp
    ).fit()
    xs = torch.rand(10, 2)
    a, b = m.predict(xs)[0], m2.predict(xs)[0]
    # level-2 sigma2 differs only through the (zero-weighted) rho*yc term -> identical data
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-8)


def test_posterior_matches_predict_and_is_psd():
    X, Y = branin_pair()
    m = ar1_cokriging(X, Y, regression="regpoly1", hyperparameters=FIXED).fit()
    xs = torch.rand(3, 4, 2)  # batch of 3 q=4 candidate sets
    post = m.posterior(xs)
    assert post.mean.shape == (3, 4, 1)
    for b in range(3):
        mu, var = m.predict(xs[b])
        np.testing.assert_allclose(post.mean[b, :, 0].numpy(), mu.numpy(), atol=1e-9)
        np.testing.assert_allclose(
            post.variance[b, :, 0].numpy(), var.numpy(), rtol=1e-6, atol=1e-6
        )
        cov = post.distribution.covariance_matrix[b].detach().numpy()
        assert np.linalg.eigvalsh(cov).min() > -1e-8


def test_works_with_botorch_acquisition():
    from botorch.acquisition.analytic import LogExpectedImprovement

    X, Y = branin_pair()
    m = ar1_cokriging(X, Y, regression="regpoly0", hyperparameters=FIXED).fit()
    acq = LogExpectedImprovement(m, best_f=Y[1].min(), maximize=False)
    val = acq(torch.rand(5, 1, 2))
    assert val.shape == (5,) and torch.isfinite(val).all()


def test_rejects_bad_inputs():
    X, Y = branin_pair()
    with pytest.raises(ValueError):
        ar1_cokriging(X + [X[1]], Y + [Y[1]]).fit()
    with pytest.raises(ValueError):
        ar1_cokriging(X, Y, correlation="nope").fit()


def test_forrester_1d_benchmark():
    hi = lambda x: (6 * x - 2) ** 2 * torch.sin(12 * x - 4)
    lo = lambda x: 0.5 * hi(x) + 10 * (x - 0.5) - 5
    X = [torch.linspace(0, 1, 20).reshape(-1, 1), torch.linspace(0, 1, 4).reshape(-1, 1)]
    Y = [lo(X[0][:, 0]), hi(X[1][:, 0])]
    m = ar1_cokriging(X, Y, correlation="matern32").fit()
    xs = torch.linspace(0, 1, 201).reshape(-1, 1)
    truth = hi(xs[:, 0])
    rmse = ((m.predict(xs)[0] - truth) ** 2).mean().sqrt()
    assert rmse < 0.5
    # co-kriging must beat kriging on the 4 high-fidelity points alone
    # (rho = 0 -> low fidelity ignored)
    m_hi = ar1_cokriging(
        X, Y, correlation="matern32", hyperparameters={**m.hyperparameters, "rho": 0.0}
    ).fit()
    rmse_hi = ((m_hi.predict(xs)[0] - truth) ** 2).mean().sqrt()
    assert rmse < rmse_hi


FIXTURE = Path(__file__).parent / "fixtures" / "cokriging_fixture.mat"


@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="run tests/fixtures/generate_cokriging.m in MATLAB to enable MATLAB parity checks",
)
def test_matches_matlab_fixture():
    import scipy.io as sio

    d = sio.loadmat(FIXTURE, squeeze_me=True, struct_as_record=False)
    X = [torch.tensor(np.atleast_2d(s)) for s in d["samples"]]
    Y = [torch.tensor(np.atleast_1d(v)) for v in d["values"]]
    theta = np.atleast_1d(d["theta"])
    hp = {"theta1": theta, "theta2": theta, "rho": float(d["rho"])}
    xs = torch.tensor(d["xs"])
    for case in np.atleast_1d(d["cases"]):
        m = ar1_cokriging(
            X, Y, correlation=str(case.corr), regression=str(case.regr), hyperparameters=hp
        ).fit()
        mean, var = m.predict(xs)
        np.testing.assert_allclose(mean.numpy(), case.mean, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(var.numpy(), case.var, rtol=1e-4, atol=1e-9)
