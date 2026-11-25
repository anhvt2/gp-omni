"""Acquisition functions, one package per problem type.

* :mod:`~gp_omni.opt.acquisition.single_objective` -- PI, EI, UCB, PE, MES, KG, PES
* :mod:`~gp_omni.opt.acquisition.multi_objective` -- expected hypervolume, weighted objective
  (not implemented yet)
* :mod:`~gp_omni.opt.acquisition.multi_fidelity` (not implemented yet)
* :mod:`~gp_omni.opt.acquisition.multi_objective_multi_fidelity` (not implemented yet)

The single-objective API is re-exported here because the BO loops use it.
"""

from gp_omni.opt.acquisition.common import (
    check_multiple_design_sites,
    negative_classifier_mse,
    negative_mse,
    predict_feasibility,
    predict_objective,
)
from gp_omni.opt.acquisition.single_objective import (
    NAMES,
    acquisition_value,
    check_acquisition_name,
    negative_acquisition,
    prepare_context,
)

__all__ = [
    "NAMES",
    "acquisition_value",
    "check_acquisition_name",
    "check_multiple_design_sites",
    "negative_acquisition",
    "negative_classifier_mse",
    "negative_mse",
    "predict_feasibility",
    "predict_objective",
    "prepare_context",
]
