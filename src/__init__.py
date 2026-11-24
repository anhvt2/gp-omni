"""gp-omni: a consolidation layer over GPyTorch and BoTorch, plus AR1 co-kriging.

Importing this package sets torch's default dtype to float64 (GP linear algebra needs it).
"""

import torch

torch.set_default_dtype(torch.float64)

from gp_omni.bo_config import BOConfig
from gp_omni.build.cokriging import ar1_cokriging, dag_cokriging
from gp_omni.build.gpc import GPC
from gp_omni.build.gpr import GPR

__all__ = ["GPC", "GPR", "BOConfig", "ar1_cokriging", "dag_cokriging"]
__version__ = "0.1.0"
