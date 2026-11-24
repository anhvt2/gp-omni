"""Compatibility imports for the phase-based GP implementation."""

from gp_omni.build.hyperparameters import check_hyperparameters as check_hyp
from gp_omni.build.hyperparameters import make_hyperparameters as make_hyp
from gp_omni.infer.moments import predict_gp as gp_predict
from gp_omni.infer.posterior import gaussian_posterior as inf_gauss_lik
from gp_omni.train.objectives import inference_method as Inference
from gp_omni.train.objectives import negative_log_evidence as gp_nlz

__all__ = ["Inference", "check_hyp", "gp_nlz", "gp_predict", "inf_gauss_lik", "make_hyp"]
