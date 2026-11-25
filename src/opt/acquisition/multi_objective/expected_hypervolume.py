"""Expected hypervolume improvement (EHVI) for Pareto-front search.

Not implemented yet. Planned: exact EHVI by box decomposition of the non-dominated region
(2 and 3 objectives), Monte Carlo beyond that. Sources to check before implementing:
Emmerich (2005) for the definition and Zhao et al. (2018, arXiv 1812.07692) for the exact
computation.
"""

NAMES: tuple[str, ...] = ()


def expected_hypervolume_improvement(*args, **kwargs):
    raise NotImplementedError("expected hypervolume improvement is not implemented yet")
