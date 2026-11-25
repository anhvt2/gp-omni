"""Analytical benchmarks for Gaussian-process regression and Bayesian optimization.

Four families provide scalar, multi-fidelity, multi-objective, and noisy targets:

- :mod:`.single_objective` -- fixed-dimensional and scalable scalar functions.
- :mod:`.multi_fidelity` -- Forrester, Currin, borehole, and welded beam.
- :mod:`.multi_objective` -- ZDT and scaled DTLZ variants.
- :mod:`.noisy_regression` -- logistic and heteroscedastic Hall-Petch regression.

Every function takes a ``(n, d)`` tensor and returns ``(n,)`` or ``(n, M)`` in torch's
default dtype. Fidelity levels run from cheapest = 0 to highest = L - 1, matching ``GPR``.
"""

from .common import BenchmarkFunction, MultiFidelityFunction, MultiObjectiveFunction

__all__ = ["BenchmarkFunction", "MultiFidelityFunction", "MultiObjectiveFunction"]
