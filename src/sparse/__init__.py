"""Sparse (variational) GPs, ported from gpml-matlab v4.2 ``apxSparse`` / ``apx``."""

from gp_omni.sparse.apx import SparseCov, apx
from gp_omni.sparse.cov import Cov
from gp_omni.sparse.gp import gp_nlz, gp_predict, make_hyp
from gp_omni.sparse.inf import inf_gauss_lik, inf_laplace
from gp_omni.sparse.init import init_inducing, kmeans_landmarks, random_subset
from gp_omni.sparse.lik import LikErf, LikGauss, LikLogistic, make_lik
from gp_omni.sparse.mean import Mean
from gp_omni.sparse.model import SparseGP
from gp_omni.sparse.omni import OmniGP
from gp_omni.sparse.train import train
from gp_omni.sparse.xu_opt import vfe_xu_opt

__all__ = [
    "Cov",
    "LikErf",
    "LikGauss",
    "LikLogistic",
    "Mean",
    "OmniGP",
    "SparseCov",
    "SparseGP",
    "apx",
    "gp_nlz",
    "gp_predict",
    "inf_gauss_lik",
    "inf_laplace",
    "init_inducing",
    "kmeans_landmarks",
    "make_hyp",
    "make_lik",
    "random_subset",
    "train",
    "vfe_xu_opt",
]
