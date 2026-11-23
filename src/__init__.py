"""gp-omni: a consolidation layer over GPyTorch and BoTorch, plus AR1 co-kriging.

Importing this package sets torch's default dtype to float64 (GP linear algebra needs it).
"""

import torch

torch.set_default_dtype(torch.float64)

from gp_omni.models.cokriging import AR1CoKriging

__all__ = ["AR1CoKriging"]
__version__ = "0.1.0"
