"""Metrics for minimization campaigns, in the original objective units."""

import numpy as np


def cumulative_regret(evaluation_values, global_min):
    """Sum f(x_i) - f* over actual evaluations, including initialization."""
    return np.cumsum(np.asarray(evaluation_values, dtype=float) - global_min)


def average_regret(evaluation_values, global_min):
    """Cumulative regret divided by the number of evaluations so far (T >= 1)."""
    cumulative = cumulative_regret(evaluation_values, global_min)
    return cumulative / np.arange(1, len(cumulative) + 1)
