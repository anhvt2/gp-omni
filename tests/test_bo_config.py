"""Shared BO configuration works independently of the scheduling strategy."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gp_omni.config
from gp_omni import bo_config
from gp_omni.opt.parallel.asyncbo import async_bo_config
from gp_omni.opt.parallel.batchbo import batch_bo_config


@pytest.mark.parametrize("scale", [1.5, [1.0, 2.0]])
def test_packaged_scale_defaults_import_and_remain_independent(scale, tmp_path):
    # Import a fresh package with edited defaults, before any dataclass is created.
    package = tmp_path / "gp_omni"
    shutil.copytree(
        Path(gp_omni.__file__).parent,
        package,
        ignore=shutil.ignore_patterns("__pycache__", "config"),
    )
    shutil.copytree(Path(gp_omni.config.__file__).parent, package / "config")
    settings = package / "config" / "bo.json"
    defaults = json.loads(settings.read_text())
    defaults["loop"]["x_scale"] = scale
    settings.write_text(json.dumps(defaults))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import numpy as np
from gp_omni import bo_config
from gp_omni.settings import load_defaults
from gp_omni.opt.parallel.asyncbo import async_bo_config
from gp_omni.opt.parallel.batchbo import batch_bo_config

expected = np.broadcast_to(load_defaults("bo")["loop"]["x_scale"], (2,))
for cls in (bo_config, async_bo_config, batch_bo_config):
    first = cls(model_name="first", x_lb=[0, 0], x_ub=[1, 1])
    second = cls(model_name="second", x_lb=[0, 0], x_ub=[1, 1])
    np.testing.assert_array_equal(first.x_scale, expected)
    first.x_scale[0] = 99
    np.testing.assert_array_equal(second.x_scale, expected)
    override = cls(model_name="override", x_lb=[0, 0], x_ub=[1, 1], x_scale=3)
    np.testing.assert_array_equal(override.x_scale, [3, 3])
    third = cls(model_name="third", x_lb=[0, 0], x_ub=[1, 1])
    np.testing.assert_array_equal(third.x_scale, expected)
""",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("config_type", [bo_config, async_bo_config, batch_bo_config])
def test_config_round_trip(config_type, tmp_path):
    config = config_type(
        model_name="toy",
        x_lb=[0, 0],
        x_ub=[1, 2],
        x_scale=2.0,
        known_constraint=lambda x: float(x[0] <= 0.5),
    )
    if config_type is batch_bo_config:
        config.acquisition = "UCB"
    path = tmp_path / "config.json"
    config.to_json(path)
    assert "known_constraint" not in json.loads(path.read_text())

    restored = config_type.from_json(path, seed=42, known_constraint=config.known_constraint)
    assert type(restored) is config_type
    np.testing.assert_array_equal(restored.x_lb, [0, 0])
    np.testing.assert_array_equal(restored.x_ub, [1, 2])
    np.testing.assert_array_equal(restored.x_scale, [2, 2])
    assert restored.seed == 42
    assert restored.batch.total == 6
    assert restored.known_constraint(np.array([0.1, 0.2])) == 1.0
    assert restored.known_constraint(np.array([0.9, 0.2])) == 0.0
    assert config_type.from_json(path).known_constraint(np.zeros(2)) == 1.0
    if config_type is batch_bo_config:
        assert restored.acquisition == "UCB"
