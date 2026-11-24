"""Batch-sequential parallel constrained Bayesian optimisation (python port of
``daceBatchSeqPar``: ``mainprog_benchR.m`` + ``bayesSrc/``)."""

from gp_omni.train.batchbo.loop import batch_bo, batch_bo_config

__all__ = ["batch_bo", "batch_bo_config"]
