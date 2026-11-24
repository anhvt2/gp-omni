"""GPC, the constraint-classifier twin of GPR: same four independent features."""

import inspect
import itertools
import time

import pytest
import torch

import gp_omni  # noqa: F401
from gp_omni import GPC, GPR

WEIGHTS = [1.0, 0.8, -0.6]
THRESHOLD = 0.8  # a point is feasible when the underlying constraint value exceeds this


def truth(x, o, l):
    s = torch.sin(3 * x[:, 0]) + torch.cos(2 * x[:, 1]) + x[:, 2] ** 2
    f = WEIGHTS[o] * s + 0.2 * o
    for k in range(1, l + 1):
        f = (0.8 + 0.1 * k) * f + 0.3 * k * torch.sin(5 * x[:, 0])
    return f


def label(x, o, l):
    return torch.where(truth(x, o, l) > THRESHOLD, 1.0, -1.0)


def problem(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs, ys, os_, fs = [], [], [], []
    for o, row in enumerate(sizes):
        for l, n in enumerate(row):
            x = torch.rand(n, 3, generator=g)
            xs.append(x)
            ys.append(label(x, o, l))
            os_.append(torch.full((n,), o))
            fs.append(torch.full((n,), l))
    return torch.cat(xs), torch.cat(ys), torch.cat(os_), torch.cat(fs)


XT = torch.rand(600, 3, generator=torch.Generator().manual_seed(99))


def accuracy(model, o, l, output=None):
    p = model.predict_proba(XT, output=output, fidelity=l)
    return float(((p >= 0.5).double() == (label(XT, o, l) > 0).double()).double().mean())


@pytest.mark.parametrize("mo,mf,sp,ad", list(itertools.product([0, 1], repeat=4)))
def test_every_combination_of_the_four_features(mo, mf, sp, ad):
    O, L = (3 if mo else 1), (2 if mf else 1)
    sizes = [[150 if l == 0 else 40 for l in range(L)] for _ in range(O)]
    x, y, o, f = problem(sizes)
    model = GPC(O, L, additive=2 if ad else None, nu=25 if sp else None)
    model.fit(x, y, o, f, max_iter=80)
    p = model.predict_proba(XT)
    assert p.shape == ((600, O) if O > 1 else (600,))
    assert torch.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    for k in range(O):
        assert accuracy(model, k, L - 1, output=k if O > 1 else None) > 0.85, (mo, mf, sp, ad, k)
    assert (model.inducing_inputs.shape == (25, 3)) if sp else (model.inducing_inputs is None)


def test_classifier_is_a_twin_of_the_regression_model():
    """Same constructor switches and the same fit/predict shape; only the likelihood differs."""
    reg, clf = inspect.signature(GPR.__init__), inspect.signature(GPC.__init__)
    assert list(reg.parameters) == list(clf.parameters)
    diff = {k for k in reg.parameters if reg.parameters[k].default != clf.parameters[k].default}
    assert diff == {"lik"}
    assert list(inspect.signature(GPR.fit).parameters) == list(
        inspect.signature(GPC.fit).parameters
    )
    # one config drives both: regression on the constraint value, classification on its sign
    cfg = {"num_outputs": 2, "num_fidelities": 2, "additive": 2, "nu": 25}
    x, _, o, f = problem([[100, 30], [100, 30]])
    value = torch.stack([truth(x[i : i + 1], int(o[i]), int(f[i]))[0] for i in range(len(x))])
    reg_model = GPR(**cfg).fit(x, value, o, f, max_iter=60)
    clf_model = GPC(**cfg).fit(x, torch.where(value > THRESHOLD, 1.0, -1.0), o, f, max_iter=60)
    assert reg_model.predict(XT)["fmu"].shape == clf_model.predict_proba(XT).shape == (600, 2)
    for k in range(2):
        assert accuracy(clf_model, k, 1, output=k) > 0.85
    agree = (
        ((reg_model.predict(XT)["fmu"] > THRESHOLD) == clf_model.predict_feasible(XT))
        .double()
        .mean()
    )
    assert float(agree) > 0.85  # the two twins describe the same feasible region


def test_multifidelity_labels_help_a_scarce_high_fidelity_classifier():
    # Isolate transfer: a well-sampled low fidelity reveals two feasible regions,
    # while six high-fidelity labels cannot resolve both on their own. Shift the
    # high-fidelity boundaries slightly so the fidelities are not interchangeable.
    xt = torch.linspace(0, 1, 601).unsqueeze(-1)

    def labels(x, phase):
        return torch.where(torch.sin(4 * torch.pi * x[:, 0] + phase) > 0, 1.0, -1.0)

    gains = []
    for seed in range(3):
        generator = torch.Generator().manual_seed(seed)
        low_x = torch.rand(80, 1, generator=generator)
        high_x = torch.rand(6, 1, generator=generator)
        x = torch.cat([low_x, high_x])
        y = torch.cat([labels(low_x, 0), labels(high_x, 0.15)])
        fidelity = torch.cat([torch.zeros(80), torch.ones(6)]).long()
        mf = GPC(num_fidelities=2).fit(x, y, fidelity=fidelity)
        hi_only = GPC().fit(high_x, labels(high_x, 0.15))
        expected = labels(xt, 0.15) > 0
        mf_accuracy = float((mf.predict_feasible(xt, fidelity=1) == expected).double().mean())
        hi_accuracy = float((hi_only.predict_feasible(xt) == expected).double().mean())
        assert mf_accuracy > 0.9, (seed, mf_accuracy)
        gains.append(mf_accuracy - hi_accuracy)
    assert sum(gains) / len(gains) > 0.15, gains


def test_correlated_constraints_help_a_data_poor_constraint():
    gains = []
    for seed in range(3):
        x, y, o, f = problem([[150], [10]], seed=seed)
        mo = GPC(num_outputs=2).fit(x, y, o, f)
        alone = GPC().fit(x[o == 1], y[o == 1])
        gains.append(accuracy(mo, 1, 0, output=1) - accuracy(alone, 1, 0))
    assert sum(gains) / 3 > 0.05, gains


def test_additive_classifier_beats_plain_ard_in_eight_dimensions():
    w = torch.randn(8, generator=torch.Generator().manual_seed(7))
    value = lambda z: sum(w[i] * torch.sin(3 * z[:, i] + i) for i in range(8))
    lab = lambda z: torch.where(value(z) > 0, 1.0, -1.0)
    gains = []
    for seed in range(3):
        g = torch.Generator().manual_seed(10 + seed)
        x, xt = torch.rand(150, 8, generator=g), torch.rand(600, 8, generator=g)
        acc = {}
        for name, add in (("add", 2), ("ard", None)):
            p = GPC(additive=add).fit(x, lab(x)).predict_proba(xt)
            acc[name] = float(((p >= 0.5).double() == (lab(xt) > 0).double()).double().mean())
        gains.append(acc["add"] - acc["ard"])
    assert sum(gains) / 3 > 0.03, gains


def test_sparse_decisions_agree_with_exact():
    x, y, o, f = problem([[100, 30], [100, 30]])
    exact = GPC(2, 2, additive=2).fit(x, y, o, f, max_iter=60)
    sparse = GPC(2, 2, additive=2, nu=40).fit(x, y, o, f, max_iter=60)
    agree = (exact.predict_feasible(XT) == sparse.predict_feasible(XT)).double().mean()
    assert float(agree) > 0.93


def test_probabilities_are_calibrated_better_than_a_constant():
    x, y, _, _ = problem([[200]])
    p = GPC().fit(x, y).predict_proba(XT)
    truth_label = (label(XT, 0, 0) > 0).double()
    brier = float(((p - truth_label) ** 2).mean())
    base_rate = float(truth_label.mean())
    assert brier < 0.5 * base_rate * (1 - base_rate)  # at least 2x better than predicting the rate
    for like in ("probit", "logistic"):
        q = GPC(lik=like).fit(x, y).predict_proba(XT)
        assert float(((q >= 0.5).double() == truth_label).double().mean()) > 0.9


def test_all_four_features_scale_to_many_points():
    x, y, o, f = problem([[4000, 1000], [4000, 1000]], seed=3)  # 10,000 labels, 4 tasks
    t = time.time()
    m = GPC(2, 2, additive=2, nu=30).fit(x, y, o, f, max_iter=10)
    assert time.time() - t < 240
    for k in range(2):
        assert accuracy(m, k, 1, output=k) > 0.85


def test_label_formats_and_validation():
    x, y, _, _ = problem([[60]])
    base = GPC().fit(x, y, max_iter=30).predict_proba(XT)
    for converted in ((y > 0), (y > 0).double()):  # bool and {0, 1}
        p = GPC().fit(x, converted, max_iter=30).predict_proba(XT)
        torch.testing.assert_close(p, base)
    with pytest.raises(ValueError):
        GPC().fit(x, torch.full_like(y, 2.0))
    with pytest.raises(ValueError):
        GPC(lik="gaussian")
    with pytest.raises(ValueError):
        GPC(lik="nonsense")
    with pytest.raises(RuntimeError):
        GPC().predict_proba(x)
    with pytest.raises(ValueError):
        GPC(num_fidelities=2).fit(x, y, fidelity=torch.full((60,), 5))
