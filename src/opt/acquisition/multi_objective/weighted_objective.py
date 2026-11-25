"""Weighted-objective acquisition: scalarise the objectives, then use a single-objective one.

Not implemented yet. Planned: weighted Chebyshev and weighted-sum scalarisations with
weights drawn per proposal, scored by any acquisition of
:mod:`~gp_omni.opt.acquisition.single_objective`.
"""

NAMES: tuple[str, ...] = ()


def weighted_objective(*args, **kwargs):
    raise NotImplementedError("weighted-objective acquisition is not implemented yet")
