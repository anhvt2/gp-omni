"""Batch-sequential parallel constrained Bayesian optimisation (python port of
``daceBatchSeqPar``: ``mainprog_benchR.m`` + ``bayesSrc/``)."""

from gp_omni.batchbo.loop import BatchSeqBO, BatchSeqBOConfig

__all__ = ["BatchSeqBO", "BatchSeqBOConfig"]
