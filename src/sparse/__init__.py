"""Sparse (variational) GPs, ported from gpml-matlab v4.2 ``apxSparse`` / ``apx``."""

from gp_omni.sparse.apx import SparseCov, apx
from gp_omni.sparse.cov import Cov
from gp_omni.sparse.gp import gp_nlz, gp_predict, make_hyp
from gp_omni.sparse.inf import inf_gauss_lik, inf_laplace
from gp_omni.sparse.lik import LikErf, LikGauss, LikLogistic, make_lik
from gp_omni.sparse.mean import Mean

__all__ = [
    "Cov",
    "LikErf",
    "LikGauss",
    "LikLogistic",
    "Mean",
    "SparseCov",
    "apx",
    "gp_nlz",
    "gp_predict",
    "inf_gauss_lik",
    "inf_laplace",
    "make_hyp",
    "make_lik",
]
