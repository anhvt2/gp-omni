"""Noisy 1-d regression problems from the monotonic-GP manuscript (``monotonicGP_25Aug22.tex``).

Only the data-generating processes are here: the noise-free truth, the noise model and the
sampling design. They are accuracy targets for an ordinary (non-monotonic) GPR.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class NoisyProblem:
    name: str
    lower: float
    upper: float
    truth: Callable[[Tensor], Tensor]
    noise_sd: Callable[[Tensor], Tensor]  # x (n,) -> sd (n,)
    num_train: int
    source: str = "monotonicGP_25Aug22"

    def sample(self, seed: int = 0) -> tuple[Tensor, Tensor]:
        g = torch.Generator().manual_seed(seed)
        x = self.lower + (self.upper - self.lower) * torch.rand(self.num_train, 1, generator=g)
        x = x.sort(0).values
        noise = self.noise_sd(x[:, 0]) * torch.randn(self.num_train, generator=g)
        return x, self.truth(x[:, 0]) + noise

    def grid(self, n: int = 200) -> Tensor:
        return torch.linspace(self.lower, self.upper, n).reshape(-1, 1)


def logistic_truth(x: Tensor) -> Tensor:
    return 1 / (1 + torch.exp(-x))


def hall_petch_truth(d: Tensor) -> Tensor:
    """Yield stress vs grain size ``d`` in micrometers (Fernandes & Vieira)."""
    return 16.47 + 0.0000288 / (1e-6 * d) ** 1.3


def hall_petch_noise_sd(d: Tensor) -> Tensor:
    """Printed variance, using the same numerical d in micrometres as the truth equation.

    The manuscript does not state a separate conversion in this noise term. Its large
    magnitude is reported as written rather than silently changing the noise model.
    """
    return torch.sqrt(2.2e10 * d**3)


def hall_petch_noise_sd_metres(d: Tensor) -> Tensor:
    """Alternative unit interpretation, explicitly converting micrometres to metres."""
    return hall_petch_noise_sd(1e-6 * d)


def noisy_logistic() -> NoisyProblem:
    return NoisyProblem(
        "noisy_logistic", -3.0, 3.0, logistic_truth, lambda x: torch.full_like(x, 0.1), 10
    )


def heteroscedastic_hall_petch(*, noise_in_metres: bool = False) -> NoisyProblem:
    return NoisyProblem(
        "hall_petch_noise_in_metres" if noise_in_metres else "hall_petch",
        15.0, 350.0, hall_petch_truth,
        hall_petch_noise_sd_metres if noise_in_metres else hall_petch_noise_sd, 20
    )


def all_noisy() -> list[NoisyProblem]:
    return [noisy_logistic(), heteroscedastic_hall_petch()]
