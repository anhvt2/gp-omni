"""Knowledge gradient, predictive entropy search, joint posterior, hedge arms, parallel e2e."""

import numpy as np
import pytest
import torch
from scipy.stats import norm
from torch.func import hessian, jacfwd

import gp_omni  # noqa: F401
from gp_omni import GPR
from gp_omni.opt.acquisition import negative_acquisition, prepare_context
from gp_omni.opt.acquisition.single_objective import kg, pes
from gp_omni.opt.parallel.asyncbo import async_bo, async_bo_config, hedge
from gp_omni.opt.parallel.batchbo import batch_bo, batch_bo_config


@pytest.fixture(scope="module")
def data_1d():
    rng = np.random.default_rng(3)
    x = np.array([0.05, 0.2, 0.35, 0.5, 0.8, 0.95])[:, None]
    y = np.sin(6 * x[:, 0]) + 0.5 * x[:, 0] + 0.05 * rng.standard_normal(len(x))
    return x, y, GPR().fit(torch.as_tensor(x), torch.as_tensor(y), max_iter=200)


# ------------------------------------------------------------------ joint posterior
def test_full_cov_matches_marginals_and_kernel(data_1d):
    _, _, gp = data_1d
    xs = torch.tensor([[0.1], [0.3], [0.6]])
    out = gp.predict(xs, full_cov=True)
    np.testing.assert_allclose(out["fcov"].diagonal().numpy(), out["fs2"].numpy(), atol=1e-10)
    np.testing.assert_allclose(out["fcov"].numpy(), out["fcov"].numpy().T)
    # far from the data the covariance returns to the prior kernel
    far = gp.predict(torch.tensor([[40.0], [41.0]]), full_cov=True)["fcov"].numpy()
    sf2 = float(torch.exp(2 * gp.hyp["cov"][-1].detach())) * float(gp._sd[0]) ** 2
    assert far[0, 0] == pytest.approx(sf2, rel=1e-6)


def test_full_cov_sparse_not_implemented():
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.random((20, 1)))
    sparse = GPR(nu=5).fit(x, torch.sin(6 * x[:, 0]), max_iter=20)
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        sparse.predict(x[:3], full_cov=True)


# ------------------------------------------------------------------------------- KG
def test_expected_max_of_lines_matches_monte_carlo():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=12), rng.normal(size=12)
    z = rng.standard_normal(2_000_000)
    expected = float(np.mean(np.max(a[:, None] + b[:, None] * z, axis=0)))
    assert kg.expected_max_of_lines(a, b) == pytest.approx(expected, abs=3e-3)


def test_expected_max_of_lines_edge_cases():
    assert kg.expected_max_of_lines([1.0], [0.7]) == pytest.approx(1.0)
    # two lines: E max(0, Z) = phi(0)
    assert kg.expected_max_of_lines([0.0, 0.0], [0.0, 1.0]) == pytest.approx(norm.pdf(0))
    # equal slopes: the higher intercept wins; a dominated line changes nothing
    assert kg.expected_max_of_lines([0.0, 1.0, -5.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_kg_nonnegative_and_zero_at_a_known_point(data_1d):
    x, y, gp = data_1d
    cand = kg.candidate_set([0.0], [1.0], x, y, np.random.default_rng(0))
    assert cand.shape[1] == 1
    values = [kg.knowledge_gradient(np.array([g]), gp, cand) for g in np.linspace(0, 1, 11)]
    assert min(values) >= 0
    assert max(values) > 0
    assert kg.knowledge_gradient(x[1], gp, cand) == pytest.approx(0.0, abs=1e-6)


def test_kg_not_implemented_for_sparse():
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.random((20, 1)))
    sparse = GPR(nu=5).fit(x, torch.sin(6 * x[:, 0]), max_iter=20)
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        prepare_context(
            "KG", sparse, [0.0], [1.0], x.numpy(), np.zeros(20), np.random.default_rng(0)
        )


# ------------------------------------------------------------------------------ PES
def test_se_derivative_blocks_match_autograd():
    rng = np.random.default_rng(0)
    d = 3
    m = pes.se_model(
        x=rng.random((4, d)), r=np.zeros(4), mean=0.0, lam=rng.uniform(0.5, 3, d), sf2=1.7,
        sn2=0.1, sd=1.0,
    )  # fmt: skip
    lam = torch.tensor(m.lam)

    def k(a, b):
        return m.sf2 * torch.exp(-0.5 * ((a - b) ** 2 * lam).sum())

    xs, xo = torch.tensor(rng.random(d)), torch.tensor(rng.random(d))
    grad, hess = jacfwd(k, 1)(xo, xs).numpy(), hessian(k, 1)(xo, xs).numpy()
    cross = pes.cross_with_observations(m, xo.numpy()[None], xs.numpy())[0]
    np.testing.assert_allclose(cross[1 : 1 + d], grad, atol=1e-12)
    for p, (i, j) in enumerate(pes._pairs(d)):
        assert cross[1 + d + p] == pytest.approx(hess[i, j], abs=1e-12)
    curv = pes.cross_with_curvature(m, xo.numpy()[None], xs.numpy())[0]
    np.testing.assert_allclose(curv[1:], np.diag(hess), atol=1e-12)

    f4 = jacfwd(jacfwd(jacfwd(jacfwd(k, 0), 0), 1), 1)(xs, xs).numpy()
    k_oo, _, k_qq = pes.prior_blocks(m, xs.numpy())
    n = len(m.x)
    for i in range(d):
        for j in range(d):
            assert k_qq[1 + i, 1 + j] == pytest.approx(f4[i, i, j, j], abs=1e-12)
    for p, (i, j) in enumerate(pes._pairs(d)):
        k_hh = k_oo[n + d + p, n + d + p] - pes.JITTER * m.sf2
        assert k_hh == pytest.approx(f4[i, j, i, j], abs=1e-12)


def test_truncated_moments_and_ep_match_sampling():
    mean, var = pes.truncated_moments(0.3, 2.0, 0.5, 1.0)
    z = np.random.default_rng(1).normal(0.3, np.sqrt(2.0), 2_000_000)
    z = z[z >= 0.5]
    assert mean == pytest.approx(z.mean(), abs=5e-3)
    assert var == pytest.approx(z.var(), abs=5e-3)
    nu, tau = pes.ep_truncations(
        np.array([0.3]), np.array([[2.0]]), np.array([0.5]), np.array([1.0])
    )
    assert (nu[0] + 0.3 / 2.0) / (tau[0] + 0.5) == pytest.approx(mean, abs=1e-6)
    assert 1 / (tau[0] + 0.5) == pytest.approx(var, abs=1e-6)


def test_pes_nonnegative_and_peaks_near_the_optimum(data_1d):
    x, y, gp = data_1d
    ctx = prepare_context("PES", gp, [0.0], [1.0], x, y, np.random.default_rng(0))
    optima = np.array([s.xs[0] for s in ctx["pes_states"]])
    assert np.all(np.abs(optima - 0.28) < 0.1)  # true maximiser of sin(6x) + x / 2 on [0, 1]
    grid = np.linspace(0, 1, 21)
    values = np.array([pes.predictive_entropy_search(g[None], gp, ctx) for g in grid])
    assert values.min() >= 0
    near, far = values[(grid > 0.2) & (grid < 0.35)], values[grid > 0.85]
    assert near.max() > 5 * far.max()


@pytest.mark.parametrize("kwargs", [{"cov": "matern_ard"}, {"cov": "se_iso"}, {"additive": 2}])
def test_pes_wrong_model_is_not_implemented(kwargs):
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.random((12, 2)))
    gp = GPR(**kwargs).fit(x, torch.sin(6 * x[:, 0]), max_iter=20)
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        prepare_context(
            "PES", gp, [0, 0], [1, 1], x.numpy(), np.zeros(12), np.random.default_rng(0)
        )


def test_pes_and_kg_need_context(data_1d):
    x, y, gp = data_1d
    for name in ("KG", "PES"):
        with pytest.raises(ValueError, match="context"):
            negative_acquisition(np.array([0.3]), gp, None, name, x, y)


# --------------------------------------------------------------------- hedge and parallel
def test_hedge_arms_validation():
    with pytest.raises(ValueError, match="acquisition function"):
        async_bo_config(model_name="m", x_lb=[0], x_ub=[1], hedge_arms=("EI", "nope"))
    with pytest.raises(ValueError, match="at least one"):
        async_bo_config(model_name="m", x_lb=[0], x_ub=[1], hedge_arms=())
    config = async_bo_config(model_name="m", x_lb=[0], x_ub=[1], hedge_arms=["EI", "KG"])
    assert config.hedge_arms == ("EI", "KG")


def test_hedge_rewards_follow_the_arms(tmp_path):
    from gp_omni.opt.config import batch_settings
    from gp_omni.opt.parallel import jobs

    jobs.write_settings(tmp_path, "m", batch_settings(1, 0, 0), 1.0)
    for i, (y, scheme) in enumerate([(0.0, None), (1.0, "KG"), (2.0, "PES"), (3.0, "KG")], 1):
        folder = tmp_path / f"m_iter{i}"
        folder.mkdir()
        (folder / "output.dat").write_text(f"{y}\n")
        if scheme:
            (folder / "acquisition_scheme.dat").write_text(scheme)
    arms = ("EI", "KG", "PES")
    assert hedge.get_rewards(tmp_path, 1, arms).tolist() == [0.0, 2.0, 1.0]
    assert len((tmp_path / "rewards.dat").read_text().split()) == 3
    rng = np.random.default_rng(0)
    picks = {hedge.get_acquisition_scheme(tmp_path, 1, np.array([0.0, 50.0, 0.0]), rng, arms)}
    assert picks == {"KG"}


def _simulation(x):
    return -np.sum((x - 0.3) ** 2), bool(x[0] < 0.8)


@pytest.mark.parametrize("acquisition", ["KG", "PES"])
def test_batch_bo_runs_end_to_end(tmp_path, acquisition):
    config = batch_bo_config(
        model_name="toy", x_lb=[0, 0], x_ub=[1, 1], exploit_size=2, explore_size=1,
        explore_classifier_size=0, max_iter=5, num_init_point=3, check_time=0.005,
        gp_max_iter=40, cma_max_evals=30, seed=1, acquisition=acquisition,
    )  # fmt: skip
    x, y = batch_bo(tmp_path, config, evaluate=_simulation).run()
    assert x[0] < 0.8
    assert y > -0.3


def test_async_hedge_with_all_new_arms_runs_end_to_end(tmp_path):
    arms = ("EI", "MES", "KG", "PES")
    config = async_bo_config(
        model_name="toy", x_lb=[0, 0], x_ub=[1, 1], exploit_size=2, explore_size=1,
        explore_classifier_size=0, max_iter=12, num_init_point=3, check_time=0.005,
        gp_max_iter=40, cma_max_evals=30, seed=3, hedge_arms=arms,
    )  # fmt: skip
    _, y = async_bo(tmp_path, config, evaluate=_simulation).run()
    assert y > -0.3
    assert len((tmp_path / "rewards.dat").read_text().split()) == len(arms)
