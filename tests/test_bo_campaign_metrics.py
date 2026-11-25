"""Cumulative regret charges every evaluation, including unsuccessful exploration."""

import numpy as np
import pytest
from bo_functions.campaign_metrics import average_regret, cumulative_regret


@pytest.mark.parametrize(
    "values,minimum,expected",
    [
        ([4, 8, 1], 1, [3, 10, 10]),
        ([-2, 1, -3], -3, [1, 5, 5]),
    ],
)
def test_cumulative_regret_includes_worse_evaluations(values, minimum, expected):
    np.testing.assert_allclose(cumulative_regret(values, minimum), expected)


def test_average_regret_uses_prefix_evaluation_count():
    np.testing.assert_allclose(average_regret([4, 8, 1], 1), [3, 5, 10 / 3])
