"""Parallel Bayesian optimisation: propose the next point after hallucinating pending ones.

Acquisition functions (:mod:`gp_omni.opt.acquisition`) are sequential. Parallelism only
decides *when* to propose (:mod:`~gp_omni.opt.parallel.batchbo` waits for a whole batch,
:mod:`~gp_omni.opt.parallel.asyncbo` refills slots one by one) and *what the model has seen*
(pending points are hallucinated with the posterior mean).
"""
