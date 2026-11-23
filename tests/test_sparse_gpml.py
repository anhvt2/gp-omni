"""Parity of gp_omni.sparse with gpml-matlab v4.2 (golden values from Octave)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io as sio
import torch

import gp_omni  # noqa: F401  (float64)
from gp_omni.sparse import (
    Cov,
    LikErf,
    LikGauss,
    LikLogistic,
    Mean,
    SparseCov,
    gp_nlz,
    gp_predict,
    inf_gauss_lik,
    inf_laplace,
    make_hyp,
)

FIX = Path(__file__).parent / "fixtures" / "sparse_gpml_fixture.mat"
pytestmark = pytest.mark.skipif(not FIX.exists(), reason="run generate_sparse_gpml.m")


@pytest.fixture(scope="module")
def fx():
    d = sio.loadmat(FIX, squeeze_me=True, struct_as_record=False)
    return SimpleNamespace(**{k: v for k, v in d.items() if not k.startswith("__")})


def T(a):
    return torch.tensor(np.asarray(a, dtype=float))


def hyp_of(**kw):
    out = make_hyp(**kw)
    for v in out.values():
        v.requires_grad_(True)
    return out


def check(got, want, rtol=1e-7, atol=1e-9):
    np.testing.assert_allclose(
        np.atleast_1d(got.detach().numpy()), np.atleast_1d(want), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("i", [0, 1, 2])  # s = 1, 0, 0.5
def test_gaussian_matches_gpml(fx, i):
    c = fx.gauss[i]
    x, y, xs, xu = T(fx.x), T(fx.yreg), T(fx.xs), T(fx.xu)
    hyp = hyp_of(mean=0.1, cov=np.log([0.4, 0.6, 1.2]), lik=np.log(0.2), xu=xu)
    cov, opt = SparseCov(Cov("SEard"), xu), {"s": float(c.s)}
    nlZ, d, post = gp_nlz(hyp, inf_gauss_lik, Mean("const"), cov, LikGauss(), x, y, opt)
    check(nlZ, c.nlZ)
    check(d["mean"], c.dmean)
    check(d["cov"], c.dcov)
    check(d["lik"], c.dlik)
    check(d["xu"], c.dxu, rtol=1e-6, atol=1e-8)
    p = gp_predict(hyp, post, Mean("const"), cov, LikGauss(), x, xs)
    for k in ("ymu", "ys2", "fmu", "fs2"):
        check(p[k], getattr(c, k), rtol=1e-6, atol=1e-9)


def test_learnable_inducing_noise_matches_gpml(fx):
    c = fx.snu
    x, y, xu = T(fx.x), T(fx.yreg), T(fx.xu)
    hyp = hyp_of(mean=0.1, cov=np.log([0.4, 0.6, 1.2]), lik=np.log(0.2), xu=xu, snu=np.log(3e-3))
    nlZ, d, _ = gp_nlz(
        hyp, inf_gauss_lik, Mean("const"), SparseCov(Cov("SEard"), xu), LikGauss(), x, y, {"s": 1.0}
    )
    check(nlZ, c.nlZ)
    check(d["snu"], c.dsnu, rtol=1e-6, atol=1e-9)
    check(d["cov"], c.dcov, rtol=1e-6, atol=1e-9)


def test_exact_matern_with_linear_mean_matches_gpml(fx):
    c = fx.exact_matern
    hyp = hyp_of(mean=[0.3, -0.2, 0.05], cov=np.log([0.5, 0.7, 1.1]), lik=np.log(0.15))
    nlZ, d, _ = gp_nlz(
        hyp,
        inf_gauss_lik,
        Mean("linear+const"),
        Cov("Maternard", 5),
        LikGauss(),
        T(fx.x),
        T(fx.yreg),
    )
    check(nlZ, c.nlZ)
    check(d["mean"], c.dmean)
    check(d["cov"], c.dcov)
    check(d["lik"], c.dlik)


def test_sparse_matern_vfe_matches_gpml(fx):
    c = fx.sparse_matern
    xu = T(fx.xu)
    hyp = hyp_of(mean=[0.3, -0.2, 0.05], cov=np.log([0.5, 0.7, 1.1]), lik=np.log(0.15), xu=xu)
    cov = SparseCov(Cov("Maternard", 5), xu)
    nlZ, d, _ = gp_nlz(
        hyp, inf_gauss_lik, Mean("linear+const"), cov, LikGauss(), T(fx.x), T(fx.yreg), {"s": 0.0}
    )
    check(nlZ, c.nlZ)
    for k, ref in (("mean", c.dmean), ("cov", c.dcov), ("lik", c.dlik)):
        check(d[k], ref, rtol=1e-6, atol=1e-9)
    check(d["xu"], c.dxu, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_laplace_classification_matches_gpml(fx, k):
    c = fx.laplace[k]
    lik = {"erf": LikErf, "logistic": LikLogistic}[str(c.lik)]()
    x, y, xs, xu = T(fx.x), T(fx.ycls), T(fx.xs), T(fx.xu)
    hyp = hyp_of(mean=0.0, cov=np.log([0.7, 0.9, 1.5]), lik=np.zeros(0), xu=xu)
    cov = SparseCov(Cov("SEard"), xu)
    opt = {"s": float(c.s), "irls_tol": 1e-13, "irls_maxit": 200}
    nlZ, d, post = gp_nlz(hyp, inf_laplace, Mean("const"), cov, lik, x, y, opt)
    check(nlZ, c.nlZ, rtol=1e-8)
    check(d["mean"], c.dmean, rtol=1e-5, atol=1e-8)
    check(d["cov"], c.dcov, rtol=1e-5, atol=1e-8)
    check(d["xu"], c.dxu, rtol=1e-5, atol=1e-7)
    p = gp_predict(hyp, post, Mean("const"), cov, lik, x, xs)
    check(p["fmu"], c.fmu, rtol=1e-6, atol=1e-9)
    check(p["fs2"], c.fs2, rtol=1e-6, atol=1e-9)
    check(p["ymu"], c.ymu, rtol=1e-6, atol=1e-9)


def test_vfe_xu_opt_matches_gpml(fx):
    from gp_omni.sparse.xu_opt import vfe_xu_opt

    c = fx.xu_opt
    hyp = make_hyp(mean=0.1, cov=np.log([0.4, 0.6, 1.2]), lik=np.log(0.2), xu=T(c.xu0))
    out, nlZ = vfe_xu_opt(hyp, Mean("const"), Cov("SEard"), T(fx.x), T(fx.yreg), T(fx.x), 6)
    check(out["xu"], c.xu, rtol=1e-8, atol=1e-10)
    assert nlZ == pytest.approx(float(c.nlZ), rel=1e-9)


def test_additive_covadd_matches_gpml(fx):
    cov = Cov("ADD", orders=[1, 3])
    e, sp = fx.add_exact, fx.add_sparse
    x, y = T(e.x), T(e.y)
    hyp = hyp_of(mean=0.2, cov=np.log([0.5, 0.8, 1.1, 0.9, 0.6]), lik=np.log(0.2))
    nlZ, d, _ = gp_nlz(hyp, inf_gauss_lik, Mean("const"), cov, LikGauss(), x, y)
    check(nlZ, e.nlZ)
    for k, ref in (("mean", e.dmean), ("cov", e.dcov), ("lik", e.dlik)):
        check(d[k], ref, rtol=1e-6, atol=1e-9)
    xu = T(sp.xu)
    hyp = hyp_of(mean=0.2, cov=np.log([0.5, 0.8, 1.1, 0.9, 0.6]), lik=np.log(0.2), xu=xu)
    scov = SparseCov(cov, xu)
    nlZ, d, post = gp_nlz(hyp, inf_gauss_lik, Mean("const"), scov, LikGauss(), x, y, {"s": 0.0})
    check(nlZ, sp.nlZ)
    check(d["cov"], sp.dcov, rtol=1e-6, atol=1e-9)
    check(d["xu"], sp.dxu, rtol=1e-6, atol=1e-8)
    p = gp_predict(hyp, post, Mean("const"), scov, LikGauss(), x, T(sp.xs))
    check(p["fmu"], sp.fmu, rtol=1e-6, atol=1e-9)
    check(p["fs2"], sp.fs2, rtol=1e-6, atol=1e-9)


def test_additive_cov_matches_brute_force_elementary_polynomials():
    import itertools

    g = torch.Generator().manual_seed(0)
    x, z = torch.rand(6, 4, generator=g), torch.rand(5, 4, generator=g)
    hyp = torch.tensor(np.log([0.5, 0.7, 0.9, 1.3, 0.8, 1.2, 0.6]))  # 4 ell + 3 sf
    got = Cov("ADD", orders=3)(hyp, x, z)
    ell, sf2 = hyp[:4].exp(), (2 * hyp[4:]).exp()
    kd = torch.exp(-0.5 * ((x[:, None, :] - z[None, :, :]) / ell) ** 2)
    want = sum(
        sf2[r - 1]
        * sum(torch.prod(kd[..., list(c)], -1) for c in itertools.combinations(range(4), r))
        for r in (1, 2, 3)
    )
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(
        Cov("ADD", orders=3)(hyp, x, "diag"), torch.diagonal(Cov("ADD", orders=3)(hyp, x))
    )
