"""Acquisition functions for several objectives (Pareto-front search).

Not implemented yet. Two families, kept apart because they need different machinery:

* :mod:`~gp_omni.opt.acquisition.multi_objective.expected_hypervolume`: Pareto-aware, scores
  the hypervolume a candidate adds.
* :mod:`~gp_omni.opt.acquisition.multi_objective.weighted_objective`: collapses the objectives
  to one with weights, then reuses :mod:`~gp_omni.opt.acquisition.single_objective`.
"""

NAMES: tuple[str, ...] = ()
