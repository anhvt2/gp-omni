"""Shared BO configuration works independently of the scheduling strategy."""

import json

import numpy as np
import pytest

from gp_omni import BOConfig
from gp_omni.asyncbo import AsyncBOConfig
from gp_omni.batchbo import BatchSeqBOConfig


@pytest.mark.parametrize("config_type", [BOConfig, AsyncBOConfig, BatchSeqBOConfig])
def test_config_round_trip(config_type, tmp_path):
    config = config_type(
        model_name="toy", x_lb=[0, 0], x_ub=[1, 2], x_scale=2.0,
        known_constraint=lambda x: float(x[0] <= 0.5),
    )
    if config_type is BatchSeqBOConfig:
        config.acquisition = "UCB"
        config.relax_alpha = None
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
    if config_type is BatchSeqBOConfig:
        assert restored.acquisition == "UCB"
        assert restored.relax_alpha is None
