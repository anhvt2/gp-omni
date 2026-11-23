"""OmniGP: every combination of the four independent features works, and each one matters."""

import itertools
import time

import pytest
import torch

import gp_omni  # noqa: F401
from gp_omni import OmniGP

WEIGHTS = [1.0, 0.8, -0.6]


def truth(x, o, l):
    """Additive base function, correlated across outputs, nested across fidelities."""
    s = torch.sin(3 * x[:, 0]) + torch.cos(2 * x[:, 1]) + x[:, 2] ** 2
    f = WEIGHTS[o] * s + 0.2 * o
    for k in range(1, l + 1):
        f = (0.8 + 0.1 * k) * f + 0.3 * k * torch.sin(5 * x[:, 0])
    return f


def problem(sizes, seed=0, noise=0.02):
    """``sizes[o][l]`` samples for every (output, fidelity)."""
    g = torch.Generator().manual_seed(seed)
    xs, ys, os_, fs = [], [], [], []
    for o, row in enumerate(sizes):
        for l, n in enumerate(row):
            x = torch.rand(n, 3, generator=g)
            xs.append(x)
            ys.append(truth(x, o, l) + noise * torch.randn(n, generator=g))
            os_.append(torch.full((n,), o))
            fs.append(torch.full((n,), l))
    return torch.cat(xs), torch.cat(ys), torch.cat(os_), torch.cat(fs)


XT = torch.rand(400, 3, generator=torch.Generator().manual_seed(99))


def nrmse(model, o, l):
    pred = model.predict(XT, output=o, fidelity=l)["fmu"]
    tr = truth(XT, o, l)
    return float(((pred - tr) ** 2).mean().sqrt() / tr.std())


@pytest.mark.parametrize("mo,mf,sp,ad", list(itertools.product([0, 1], repeat=4)))
def test_every_combination_of_the_four_features(mo, mf, sp, ad):
    O, L = (3 if mo else 1), (2 if mf else 1)
    sizes = [[120 if l == 0 else 30 for l in range(L)] for _ in range(O)]
    x, y, o, f = problem(sizes)
    model = OmniGP(O, L, additive=2 if ad else None, nu=25 if sp else None)
    model.fit(x, y, o, f, max_iter=120)
    pred = model.predict(XT)
    shape = (400, O) if O > 1 else (400,)
    assert pred["fmu"].shape == shape and pred["fs2"].shape == shape
    assert torch.isfinite(pred["fmu"]).all() and (pred["fs2"] >= 0).all()
    assert (pred["ys2"] >= pred["fs2"] - 1e-12).all()
    for k in range(O):
        assert nrmse(model, k, L - 1) < 0.1, (mo, mf, sp, ad, k)
    if sp:
        assert model.inducing_inputs.shape == (25, 3)
    else:
        assert model.inducing_inputs is None


def test_multifidelity_beats_high_fidelity_only():
    x, y, o, f = problem([[150, 8]])
    mf = OmniGP(num_fidelities=2).fit(x, y, o, f)
    hi = f == 1
    single = OmniGP().fit(x[hi], y[hi])
    assert nrmse(mf, 0, 1) < 0.5 * nrmse(single, 0, 0), (nrmse(mf, 0, 1), nrmse(single, 0, 0))


def test_correlated_outputs_help_a_data_poor_output():
    x, y, o, f = problem([[150], [8]])
    mo = OmniGP(num_outputs=2).fit(x, y, o, f)
    poor = o == 1
    single = OmniGP().fit(x[poor], y[poor])
    assert nrmse(mo, 1, 0) < 0.5 * nrmse(single, 0, 0), (nrmse(mo, 1, 0), nrmse(single, 0, 0))


def test_additive_beats_plain_ard_in_eight_dimensions():
    g = torch.Generator().manual_seed(7)
    w = torch.randn(8, generator=g)
    fn = lambda x: sum(w[i] * torch.sin(3 * x[:, i] + i) for i in range(8))
    x, xt = torch.rand(120, 8, generator=g), torch.rand(300, 8, generator=g)
    y = fn(x) + 0.02 * torch.randn(120, generator=g)
    err = {}
    for name, add in (("add", 2), ("ard", None)):
        m = OmniGP(additive=add).fit(x, y)
        err[name] = float(((m.predict(xt)["fmu"] - fn(xt)) ** 2).mean().sqrt() / fn(xt).std())
    assert err["add"] < 0.5 * err["ard"], err


def test_sparse_close_to_exact_and_independent_of_other_switches():
    x, y, o, f = problem([[100, 25], [100, 25]])
    exact = OmniGP(2, 2, additive=2).fit(x, y, o, f, max_iter=100)
    sparse = OmniGP(2, 2, additive=2, nu=40).fit(x, y, o, f, max_iter=100)
    a, b = exact.predict(XT)["fmu"], sparse.predict(XT)["fmu"]
    assert float(((a - b) ** 2).mean().sqrt() / a.std()) < 0.05


def test_all_four_features_scale_to_many_points():
    x, y, o, f = problem([[6000, 1500], [6000, 1500]], seed=3)  # 15,000 points, 4 tasks
    t = time.time()
    m = OmniGP(2, 2, additive=2, nu=30).fit(x, y, o, f, max_iter=15)
    elapsed = time.time() - t
    assert elapsed < 180, elapsed
    for k in range(2):
        assert nrmse(m, k, 1) < 0.1


def test_classification_works_with_sparse_and_additive():
    g = torch.Generator().manual_seed(4)
    rule = lambda z: torch.where(torch.sin(3 * z[:, 0]) + z[:, 1] ** 2 > 0.6, 1.0, -1.0)
    x, xt = torch.rand(300, 3, generator=g), torch.rand(400, 3, generator=g)
    m = OmniGP(additive=2, nu=20, lik="likErf").fit(x, rule(x), max_iter=60)
    assert float((torch.sign(m.predict(xt)["ymu"]) == rule(xt)).double().mean()) > 0.9


def test_input_validation():
    x, y, o, f = problem([[20, 5]])
    with pytest.raises(ValueError):
        OmniGP(num_fidelities=2, lik="nonsense")
    OmniGP(num_fidelities=2, lik="likErf")  # classification is allowed with every feature
    with pytest.raises(ValueError):
        OmniGP(num_fidelities=2).fit(x, y, o, f + 1)  # fidelity index out of range
    with pytest.raises(RuntimeError):
        OmniGP().predict(x)
