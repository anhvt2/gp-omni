"""Acquisition functions for one objective observed at several fidelities.

Not implemented yet. Planned: multi-fidelity max-value entropy search (Takeno et al., ICML
2020, arXiv 1901.08275), which picks ``(x, fidelity)`` by information gain about the optimal
value per unit cost, using the co-kriging models in :mod:`gp_omni.build.cokriging`.
"""

NAMES: tuple[str, ...] = ()
