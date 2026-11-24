"""Construction, optimization and prediction expose separate, usable phases."""

import importlib

import pytest
import torch

from gp_omni import build, infer, train
from gp_omni.build import GPC, GPR, ar1_cokriging, dag_cokriging, sparse_gp
from gp_omni.build.likelihood import gaussian, logistic, make_likelihood, probit


@pytest.mark.parametrize("model_type", [GPR, GPC, sparse_gp])
def test_build_train_infer_lifecycle(model_type):
    x = torch.linspace(0, 1, 16).unsqueeze(-1)
    y = x[:, 0] > 0.5 if model_type is GPC else torch.sin(6 * x[:, 0])
    model = model_type(nu=5)
    assert model.hyp is None
    with pytest.raises(RuntimeError, match="fit"):
        infer.predict(model, x[:3])
    assert train.fit(model, x, y, max_iter=4) is model
    parameters = {key: value.clone() for key, value in model.hyp.items()}
    prediction = infer.predict(model, x[:3])
    assert prediction["fmu"].shape == (3,)
    assert torch.isfinite(prediction["fmu"]).all()
    for key, value in parameters.items():
        torch.testing.assert_close(model.hyp[key], value)
    # Importing the numerical moment implementation must not shadow infer.predict.
    importlib.import_module("gp_omni.infer.moments")
    assert callable(infer.predict)
    assert build.GPR is GPR


@pytest.mark.parametrize("model_type", [ar1_cokriging, dag_cokriging])
def test_cokriging_construction_does_not_optimize(model_type, monkeypatch):
    trainer = importlib.import_module("gp_omni.train.cokriging")
    calls = []

    def optimizer(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("optimizer reached")

    monkeypatch.setattr(trainer, "_minimize", optimizer)
    x = [torch.linspace(0, 1, 8).unsqueeze(-1), torch.linspace(0, 1, 4).unsqueeze(-1)]
    y = [torch.sin(5 * level[:, 0]) for level in x]
    args = (x, y, [(0, 1)]) if model_type is dag_cokriging else (x, y)
    model = model_type(*args, num_restarts=0)
    assert not calls
    for operation in (infer.predict, lambda model, x: model.posterior(x)):
        with pytest.raises(RuntimeError, match="fit"):
            operation(model, x[1])
    with pytest.raises(RuntimeError, match="optimizer reached"):
        train.fit(model)
    assert calls == [True]
    assert not model._fitted


@pytest.mark.parametrize(
    "name,expected", [("gaussian", gaussian), ("probit", probit), ("logistic", logistic)]
)
def test_likelihood_names_select_the_expected_model(name, expected):
    assert isinstance(make_likelihood(name), expected)
    model = GPR(lik=name)
    x = torch.linspace(0, 1, 12).unsqueeze(-1)
    y = torch.sin(6 * x[:, 0]) if name == "gaussian" else x[:, 0] > 0.5
    train.fit(model, x, y.to(torch.float64) * 2 - 1 if name != "gaussian" else y, max_iter=3)
    assert isinstance(model.lik, expected)
    assert torch.isfinite(infer.predict(model, x[:2])["fmu"]).all()
