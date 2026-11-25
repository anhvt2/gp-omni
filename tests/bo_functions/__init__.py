"""Numerical test functions ported from the author's Bayesian-optimization manuscripts.

Only the functions are here (no BO algorithm). Four families, one module each:

- :mod:`.single_objective`  -- analytic single-fidelity benchmarks (async/parallel BO paper).
- :mod:`.multi_fidelity`    -- Forrester, Currin, borehole, welded beam (multi-fidelity BO paper).
- :mod:`.multi_objective`   -- ZDT and DTLZ suites (multi-objective BO papers).
- :mod:`.noisy_regression`  -- noisy logistic and heteroscedastic Hall-Petch (monotonic GP paper).

Every function takes a ``(n, d)`` tensor and returns ``(n,)`` (or ``(n, M)``) in torch's default
dtype. Fidelity levels are indexed cheapest = 0 ... highest = L - 1, matching ``GPR``.
"""

from .common import BenchmarkFunction, MultiFidelityFunction, MultiObjectiveFunction

__all__ = ["BenchmarkFunction", "MultiFidelityFunction", "MultiObjectiveFunction"]
