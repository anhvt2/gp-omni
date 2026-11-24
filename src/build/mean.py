"""gpml mean functions (``meanZero``, ``meanConst``, ``meanLinear`` and their sum)."""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("models")

_KINDS = ("zero", "const", "linear", "linear+const")


class mean_function:
    """``zero`` (no hyp), ``const`` (``[c]``), ``linear`` (``[a_1..a_D]``) or
    ``linear+const`` (``[a_1..a_D, c]``, gpml ``{@meanSum,{@meanLinear,@meanConst}}``)."""

    def __init__(self, kind: str = DEFAULTS["mean"]["kind"]):
        if kind not in _KINDS:
            raise ValueError(f"unknown mean {kind!r}; choose from {_KINDS}")
        self.kind = kind

    def num_hyp(self, dim: int) -> int:
        return {"zero": 0, "const": 1, "linear": dim, "linear+const": dim + 1}[self.kind]

    def __call__(self, hyp: Tensor, x: Tensor) -> Tensor:
        n = x.shape[0]
        if self.kind == "zero":
            return torch.zeros(n, dtype=x.dtype)
        if self.kind == "const":
            return hyp[0] * torch.ones(n, dtype=x.dtype)
        if self.kind == "linear":
            return x @ hyp
        return x @ hyp[:-1] + hyp[-1]
