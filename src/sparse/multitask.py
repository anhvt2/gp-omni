"""Independent multi-output and multi-fidelity structure, as one gpml-style covariance.

Tasks are ``(output o, fidelity l)`` pairs. Inputs carry two extra trailing columns
``[..., o, l]`` (see :func:`augment`), so the covariance, mean and likelihood below plug into
:mod:`gp_omni.sparse` inference unchanged, exact or sparse, with any base covariance
(including the additive ``ADD``).

Model (autoregressive, Kennedy & O'Hagan 2000, generalised to any number of fidelities
and correlated outputs)::

    f_{o,0} = g_0[o]
    f_{o,l} = rho[o,l] * f_{o,l-1} + g_l[o],      l = 1 .. L-1
    g_q ~ GP(0, B_q (x) k_q),                     B_q = W_q W_q' + diag(kappa_q^2)

so ``cov(f_{o,l}(x), f_{o',l'}(x')) = sum_q c_q(o,l) c_q(o',l') B_q[o,o'] k_q(x,x')`` with
``c_q(o,l) = prod_{j=q+1..l} rho[o,j]`` (0 for ``q > l``). With one output this is the AR1
co-kriging covariance; with one fidelity it is a rank-``r`` ICM/LMC. Each feature is switched
on independently: ``num_outputs = 1`` removes ``B_q``, ``num_fidelities = 1`` removes ``rho``.

The hyperparameter vector is laid out as, for every latent ``q`` in ``0..L-1``,
``[base_q hyp, W_q (O*r), log kappa_q (O)]`` (the last two only if ``O > 1``), followed by
``rho`` (``O*(L-1)``, row-major over ``(o, l-1)``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from gp_omni.sparse.cov import Cov
from gp_omni.sparse.lik import LikGauss
from gp_omni.sparse.mean import Mean


def augment(x: Tensor, output: Tensor | None = None, fidelity: Tensor | None = None) -> Tensor:
    """Append the (output, fidelity) task columns to ``x`` (defaults: 0)."""
    n = x.shape[0]
    o = torch.zeros(n, dtype=x.dtype) if output is None else torch.as_tensor(output).to(x.dtype)
    f = torch.zeros(n, dtype=x.dtype) if fidelity is None else torch.as_tensor(fidelity).to(x.dtype)
    return torch.cat([x, o.reshape(-1, 1), f.reshape(-1, 1)], dim=1)


def _split(x: Tensor, dim: int) -> tuple[Tensor, Tensor, Tensor]:
    return x[:, :dim], x[:, dim].round().long(), x[:, dim + 1].round().long()


def task_index(x: Tensor, dim: int, num_fidelities: int) -> Tensor:
    """Flat task index ``o * L + l`` of every row of an augmented ``x``."""
    _, o, f = _split(x, dim)
    return o * num_fidelities + f


class TaskCov:
    """Structured multi-output / multi-fidelity covariance (see module docstring)."""

    def __init__(
        self, base: Cov, dim: int, num_outputs: int = 1, num_fidelities: int = 1, rank: int = 1
    ):
        if num_outputs < 1 or num_fidelities < 1:
            raise ValueError("need at least one output and one fidelity")
        self.base, self.dim, self.O, self.L = base, dim, num_outputs, num_fidelities
        self.rank = min(rank, num_outputs)
        self.nb = base.num_hyp(dim)

    @property
    def num_tasks(self) -> int:
        return self.O * self.L

    def _per_latent(self) -> int:
        return self.nb + (self.O * self.rank + self.O if self.O > 1 else 0)

    def num_hyp(self, dim: int | None = None) -> int:
        return self.L * self._per_latent() + self.O * (self.L - 1)

    def initial_hyp(self, base_hyp: Tensor, w0: float = 0.7, kappa0: float = 0.5) -> Tensor:
        """A reasonable starting vector: same base hyperparameters for every latent process."""
        parts: list[Tensor] = []
        for _ in range(self.L):
            parts.append(base_hyp.clone())
            if self.O > 1:
                parts += [
                    torch.full((self.O * self.rank,), w0),
                    torch.full((self.O,), float(torch.log(torch.tensor(kappa0)))),
                ]
        parts.append(torch.ones(self.O * (self.L - 1)))
        return torch.cat(parts)

    def _unpack(self, hyp: Tensor):
        bases, Bs, i = [], [], 0
        for _ in range(self.L):
            bases.append(hyp[i : i + self.nb])
            i += self.nb
            if self.O > 1:
                W = hyp[i : i + self.O * self.rank].reshape(self.O, self.rank)
                i += self.O * self.rank
                kap = torch.exp(2 * hyp[i : i + self.O])
                i += self.O
                Bs.append(W @ W.T + torch.diag(kap))
            else:
                Bs.append(torch.ones(1, 1, dtype=hyp.dtype))
        rho = hyp[i:].reshape(self.O, self.L - 1)
        return bases, Bs, rho

    def loadings(self, rho: Tensor) -> Tensor:
        """``C[q, o, l] = c_q(o, l)``: how latent process ``q`` enters task ``(o, l)``."""
        per_q = []
        for q in range(self.L):
            cols = []
            c = torch.zeros(self.O, dtype=rho.dtype)
            for l in range(self.L):
                if l < q:
                    c = torch.zeros(self.O, dtype=rho.dtype)
                elif l == q:
                    c = torch.ones(self.O, dtype=rho.dtype)
                else:
                    c = c * rho[:, l - 1]  # c_q(o, l) = c_q(o, l-1) * rho[o, l]
                cols.append(c)
            per_q.append(torch.stack(cols, dim=1))
        return torch.stack(per_q)

    def __call__(self, hyp: Tensor, x: Tensor, z: Tensor | str | None = None) -> Tensor:
        bases, Bs, rho = self._unpack(hyp)
        C = self.loadings(rho)
        xf, ox, lx = _split(x, self.dim)
        terms = []
        for q in range(self.L):
            cx = C[q][ox, lx]
            if isinstance(z, str):
                terms.append(cx**2 * Bs[q][ox, ox] * self.base(bases[q], xf, "diag"))
                continue
            if z is None:
                zf, oz, lz = xf, ox, lx
            else:
                zf, oz, lz = _split(z, self.dim)
            kq = self.base(bases[q], xf, None if z is None else zf)
            terms.append((cx[:, None] * C[q][oz, lz][None, :]) * Bs[q][ox][:, oz] * kq)
        return torch.stack(terms).sum(0)


class TaskMean(Mean):
    """One constant mean per task (``hyp`` has ``O*L`` entries)."""

    def __init__(self, dim: int, num_outputs: int, num_fidelities: int):
        self.dim, self.L, self.T = dim, num_fidelities, num_outputs * num_fidelities
        self.kind = "task_const"

    def num_hyp(self, dim: int | None = None) -> int:
        return self.T

    def __call__(self, hyp: Tensor, x: Tensor) -> Tensor:
        return hyp[task_index(x, self.dim, self.L)]


class LikGaussTask(LikGauss):
    """Gaussian noise with one standard deviation per task (``hyp = [log sn_t]``)."""

    kind = "likGaussTask"

    def __init__(self, dim: int, num_outputs: int, num_fidelities: int):
        self.dim, self.L, self.T = dim, num_fidelities, num_outputs * num_fidelities

    def num_hyp(self) -> int:
        return self.T

    def noise_var(self, hyp: Tensor, x: Tensor) -> Tensor:
        return torch.exp(2 * hyp[task_index(x, self.dim, self.L)])

    def laplace(self, hyp: Tensor, y: Tensor, f: Tensor):  # pragma: no cover - Gaussian only
        raise NotImplementedError("per-task noise is only used with infGaussLik")


def task_expand(dim: int, num_outputs: int, num_fidelities: int):
    """Sparse-inducing-input expansion: every location ``xu`` serves every task."""
    pairs = [(o, l) for o in range(num_outputs) for l in range(num_fidelities)]

    def expand(xu: Tensor) -> Tensor:
        n = xu.shape[0]
        return torch.cat(
            [augment(xu, torch.full((n,), float(o)), torch.full((n,), float(l))) for o, l in pairs]
        )

    return expand
