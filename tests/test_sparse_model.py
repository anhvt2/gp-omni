"""Properties of the sparse GP that do not depend on reference fixtures."""

import time

import numpy as np
import pytest
import torch

import gp_omni  # noqa: F401
from gp_omni.build.covariance import covariance
from gp_omni.build.hyperparameters import make_hyperparameters
from gp_omni.build.inducing import init_inducing
from gp_omni.build.likelihood import gaussian
from gp_omni.build.mean import mean_function
from gp_omni.build.sparse import sparse_gp
from gp_omni.build.sparse_covariance import sparse_covariance
from gp_omni.infer.covariance import apx
from gp_omni.infer.moments import predict_gp
from gp_omni.infer.posterior import gaussian_posterior
from gp_omni.train.objectives import negative_log_evidence
from gp_omni.train.optimize import optimize

FN = lambda x: torch.sin(6 * x[:, 0]) + x[:, 1] ** 2


def data(n=120, seed=0, noise=0.1):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g)
    return x, FN(x) + noise * torch.randn(n, generator=g)


def hyp0(xu=None):
    return make_hyperparameters(
        mean=0.0, cov=np.log([0.4, 0.6, 1.0]), lik=np.log(0.2), **({} if xu is None else {"xu": xu})
    )


def test_inducing_equal_to_training_recovers_exact_gp():
    x, y = data(60)
    h = hyp0()
    nlz_exact, _, post_e = negative_log_evidence(
        h,
        gaussian_posterior,
        mean_function("const"),
        covariance("se_ard"),
        gaussian(),
        x,
        y,
        grad=False,
    )
    cov = sparse_covariance(covariance("se_ard"), x.clone())
    for s in (0.0, 1.0):
        nlz, _, post = negative_log_evidence(
            h,
            gaussian_posterior,
            mean_function("const"),
            cov,
            gaussian(),
            x,
            y,
            {"s": s},
            grad=False,
        )
        assert float(nlz) == pytest.approx(float(nlz_exact), abs=1e-3)
    xs = torch.rand(20, 2)
    pe = predict_gp(h, post_e, mean_function("const"), covariance("se_ard"), gaussian(), x, xs)
    ps = predict_gp(h, post, mean_function("const"), cov, gaussian(), x, xs)
    np.testing.assert_allclose(ps["fmu"].numpy(), pe["fmu"].numpy(), atol=1e-4)
    np.testing.assert_allclose(ps["fs2"].numpy(), pe["fs2"].numpy(), atol=1e-4)


def test_vfe_is_a_lower_bound_on_the_exact_log_marginal_likelihood():
    x, y = data(80)
    h = hyp0()
    exact, _, _ = negative_log_evidence(
        h,
        gaussian_posterior,
        mean_function("const"),
        covariance("se_ard"),
        gaussian(),
        x,
        y,
        grad=False,
    )
    for nu in (5, 15, 40):
        xu = init_inducing(x, nu, "random", seed=nu)
        vfe, _, _ = negative_log_evidence(
            h,
            gaussian_posterior,
            mean_function("const"),
            sparse_covariance(covariance("se_ard"), xu),
            gaussian(),
            x,
            y,
            {"s": 0.0},
            grad=False,
        )
        assert float(vfe) >= float(exact) - 1e-9  # -ELBO >= -log p(y)


def test_vfe_bound_tightens_as_inducing_inputs_are_added():
    x, y = data(100)
    h = hyp0()
    vals = []
    for nu in (4, 12, 30, 100):
        xu = x[:nu].clone()
        v, _, _ = negative_log_evidence(
            h,
            gaussian_posterior,
            mean_function("const"),
            sparse_covariance(covariance("se_ard"), xu),
            gaussian(),
            x,
            y,
            {"s": 0.0},
            grad=False,
        )
        vals.append(float(v))
    assert vals == sorted(vals, reverse=True)


def test_training_improves_marginal_likelihood_and_learns_inducing_inputs():
    x, y = data(150)
    xu0 = init_inducing(x, 10, "kmeans", seed=1)
    h0 = hyp0(xu0.clone())
    for v in h0.values():
        v.requires_grad_(True)
    cov, opt = sparse_covariance(covariance("se_ard"), xu0), {"s": 0.0}
    nlz0, _, _ = negative_log_evidence(
        h0, gaussian_posterior, mean_function("const"), cov, gaussian(), x, y, opt, grad=False
    )
    h1, nlz1 = optimize(
        h0, gaussian_posterior, mean_function("const"), cov, gaussian(), x, y, opt, max_iter=100
    )
    assert nlz1 < float(nlz0) - 1.0
    assert not torch.allclose(h1["xu"], xu0)  # inducing inputs moved
    fixed_xu, nlz_fixed = optimize(
        h0,
        gaussian_posterior,
        mean_function("const"),
        cov,
        gaussian(),
        x,
        y,
        opt,
        max_iter=100,
        fixed=("xu",),
    )
    assert torch.allclose(fixed_xu["xu"], xu0)
    assert nlz1 <= nlz_fixed + 1e-6  # optimising xu cannot be worse


def test_kmeans_landmarks_beat_random_subsets_on_nystrom_error():
    x, _ = data(400, seed=3)
    h = hyp0()
    cov = covariance("se_ard")

    def trace_err(xu):
        K = apx(dict(h, xu=xu), sparse_covariance(cov, xu), x, {"s": 1.0})
        return float(K.extra["g"].sum())  # trace(K - Q): Nystrom approximation error

    err_km = np.mean([trace_err(init_inducing(x, 12, "kmeans", seed=s)) for s in range(8)])
    err_rd = np.mean([trace_err(init_inducing(x, 12, "random", seed=s)) for s in range(8)])
    assert err_km < err_rd


def test_inducing_noise_hyperparameter_is_differentiable():
    x, y = data(50)
    xu = init_inducing(x, 8, "random")
    h = hyp0(xu)
    h["snu"] = torch.tensor([np.log(1e-2)], requires_grad=True)
    h["cov"].requires_grad_(True)
    _, d, _ = negative_log_evidence(
        h,
        gaussian_posterior,
        mean_function("const"),
        sparse_covariance(covariance("se_ard"), xu),
        gaussian(),
        x,
        y,
        {"s": 1.0},
    )
    assert torch.isfinite(d["snu"]).all() and d["snu"].abs().sum() > 0


@pytest.mark.parametrize("s", [0.0, 1.0])
def test_sparsegp_regression_end_to_end(s):
    x, y = data(600)
    xt = torch.rand(300, 2, generator=torch.Generator().manual_seed(9))
    m = sparse_gp(25, s=s).fit(x, y, max_iter=150)
    p = m.predict(xt)
    rmse = float(((p["fmu"] - FN(xt)) ** 2).mean().sqrt())
    assert rmse < 0.05
    assert (p["fs2"] >= 0).all() and (p["ys2"] > p["fs2"]).all()
    if s == 0.0:  # VFE does not underestimate the noise (true sd = 0.1)
        assert m.hyp["lik"].exp().item() == pytest.approx(0.1, rel=0.2)


def test_sparsegp_swap_then_train_runs_and_is_not_worse():
    x, y = data(200)
    base = sparse_gp(8, init="random", seed=2).fit(x, y, learn_xu=False, max_iter=60)
    swapped = sparse_gp(8, init="random", seed=2).fit(x, y, learn_xu=False, swap=8, max_iter=60)
    assert swapped.nlZ <= base.nlZ + 1e-6


@pytest.mark.parametrize("lik", ["probit", "logistic"])
def test_sparsegp_classification(lik):
    g = torch.Generator().manual_seed(4)
    rule = lambda x: torch.where(x[:, 0] + x[:, 1] > 1, 1.0, -1.0)
    x, xt = torch.rand(300, 2, generator=g), torch.rand(400, 2, generator=g)
    m = sparse_gp(20, lik=lik, s=0.0).fit(x, rule(x), max_iter=80)
    p = m.predict(xt)
    acc = float((torch.sign(p["ymu"]) == rule(xt)).double().mean())
    assert acc > 0.9
    assert ((p["ymu"] >= -1) & (p["ymu"] <= 1)).all()


def test_sparsegp_rejects_bad_labels():
    x, y = data(30)
    with pytest.raises(ValueError):
        sparse_gp(5, lik="probit").fit(x, y)


def test_scales_to_twenty_thousand_points():
    g = torch.Generator().manual_seed(5)
    n = 20_000
    x = torch.rand(n, 2, generator=g)
    y = FN(x) + 0.1 * torch.randn(n, generator=g)
    t = time.time()
    m = sparse_gp(50).fit(x, y, max_iter=25)
    elapsed = time.time() - t
    xt = torch.rand(500, 2, generator=g)
    rmse = float(((m.predict(xt)["fmu"] - FN(xt)) ** 2).mean().sqrt())
    assert rmse < 0.05
    assert elapsed < 120, elapsed  # O(n nu^2): an exact GP would need a 20000^2 matrix
