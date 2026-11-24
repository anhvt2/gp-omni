"""Packaged settings and per-run model configuration work through prediction."""

import json

import pytest
import torch

from gp_omni import GPC, GPR
from gp_omni.build.sparse import sparse_gp
from gp_omni.settings import load_defaults


@pytest.mark.parametrize("model_type", [GPR, GPC, sparse_gp])
def test_model_config_fit_predict(model_type, tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"nu": 5, "cov": "matern_ard", "seed": 12}))
    # An explicit override takes precedence; omitted arguments retain model defaults.
    model = model_type.from_json(path, seed=7)
    assert model.seed == 7
    assert model.nu == 5
    if model_type is sparse_gp:
        assert model.cov.kind == "matern_ard"
    else:
        assert model.cov_kind == "matern_ard"
    if model_type is GPC:
        assert model.lik_name == "probit"

    x = torch.linspace(0, 1, 12).unsqueeze(-1)
    y = x[:, 0] > 0.5 if model_type is GPC else torch.sin(6 * x[:, 0])
    model.fit(x, y, max_iter=3, learn_xu=False)
    result = model.predict(x[:3])
    assert result["fmu"].shape == (3,)
    for key in ("fmu", "fs2", "ymu", "ys2"):
        assert torch.isfinite(result[key]).all()
    if model_type is GPC:
        probability = model.predict_proba(x[:3])
        assert ((probability >= 0) & (probability <= 1)).all()


@pytest.mark.parametrize("model_type", [GPR, GPC, sparse_gp])
def test_model_config_rejects_unknown_keys(model_type, tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"nu": 3, "typo": 42}')
    with pytest.raises(TypeError, match="typo"):
        model_type.from_json(path)


def test_model_config_requires_object(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("[]")
    with pytest.raises(TypeError, match="JSON object"):
        GPR.from_json(path)


def test_packaged_defaults_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    values = load_defaults("models")
    values["gpr"]["cov"] = "changed"
    assert load_defaults("models")["gpr"]["cov"] == "se_ard"
    for name in ("training", "inference", "builders", "bo"):
        assert load_defaults(name)
    with pytest.raises(ValueError, match="unknown defaults"):
        load_defaults("missing")
