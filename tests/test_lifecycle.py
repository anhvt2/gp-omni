"""Construction, optimization and prediction expose separate, usable phases."""

import importlib

import pytest
import torch

from gp_omni import build, infer, train
from gp_omni.build import GPC, GPR, sparse_gp
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
