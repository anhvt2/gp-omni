"""gpml's ``vfe_xu_opt.m``: greedy swapping of inducing inputs for the VFE bound.

Starting from the inducing inputs in ``hyp['xu']`` and a pool of candidates ``z`` (typically
the training inputs), each round evaluates the VFE negative log marginal likelihood and, in
closed form, the change from removing each inducing input (``du``) and from adding each
candidate (``dz``). The most helpful removal/addition pair is swapped. The loop stops as
soon as a swap does not improve ``nlZ``, or after ``nswap`` rounds.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gp_omni.build.covariance import covariance
from gp_omni.build.mean import mean_function
from gp_omni.settings import load_defaults

DEFAULTS = load_defaults("inference")


def _vfe(
    hyp: dict, mean: mean_function, cov: covariance, xu: Tensor, x: Tensor, y: Tensor, z: Tensor
):
    n, nu = x.shape[0], xu.shape[0]
    hc = hyp["cov"]
    m = mean(hyp["mean"], x)
    sn2 = torch.exp(2 * hyp["lik"][0])
    Kuu, Ku, diagK = cov(hc, xu), cov(hc, xu, x), cov(hc, x, "diag")
    snu2 = DEFAULTS["sparse"]["inducing_noise_variance_ratio"] * (
        torch.trace(Kuu) / nu
    )  # stabilise by 0.1% of signal std
    eye = torch.eye(nu, dtype=x.dtype)
    Luu = torch.linalg.cholesky(Kuu + snu2 * eye, upper=True)  # Kuu + snu2*I = Luu'*Luu
    inv_Luu = torch.linalg.solve_triangular(Luu, eye, upper=True)
    dgiKuu = (inv_Luu**2).sum(1)  # diag(inv(Kuu))
    V = torch.linalg.solve_triangular(Luu.T, Ku, upper=False)  # V'*V = Q
    Lu = torch.linalg.cholesky(eye + V @ V.T / sn2, upper=True)
    r = y - m
    R = torch.linalg.solve_triangular(Luu, V, upper=True)
    B = torch.linalg.solve_triangular(Lu.T, V, upper=False)
    alpha = r / sn2 - B.T @ (B @ r) / sn2**2
    t = torch.clamp(diagK - (V * V).sum(0), min=0.0) / sn2
    nlZ = (
        r @ alpha / 2
        + torch.log(torch.diagonal(Lu)).sum()
        + t.sum() / 2
        + n * torch.log(2 * math.pi * sn2) / 2
    )

    q = R / sn2 - (R @ B.T) @ B / sn2**2
    Rq = (R * q).sum(1)
    du = (
        (R @ alpha) ** 2 / (dgiKuu - Rq) / 2  # aKa
        + torch.log(1 - Rq / dgiKuu) / 2  # logdet Lui
        + (R * R).sum(1) / dgiKuu / sn2 / 2  # dst2
    )
    Kzu, Kz, diagKz = cov(hc, z, xu), cov(hc, x, z), cov(hc, z, "diag")
    Rz = Kzu @ R - Kz.T
    qz = Rz / sn2 - (Rz @ B.T) @ B / sn2**2
    Rqz = (Rz * qz).sum(1)
    c = torch.clamp(
        diagKz - (torch.linalg.solve_triangular(Luu, Kzu, upper=True, left=False) ** 2).sum(1),
        min=0.0,
    )
    dz = (
        -((Rz @ alpha) ** 2) / (c + Rqz) / 2
        + torch.log(1 + Rqz / c) / 2
        - (Rz * Rz).sum(1) / c / sn2 / 2
    )
    return nlZ, du, dz


def vfe_xu_opt(
    hyp: dict,
    mean: mean_function,
    cov: covariance,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    nswap: int,
    verbose: bool = False,
) -> tuple[dict, float]:
    """Greedy VFE inducing-input swaps; returns ``(hyp_with_new_xu, nlZ)``."""
    hd = {k: v.detach() for k, v in hyp.items()}
    u, z = hd["xu"].clone(), z.detach().clone()
    nlZmin, umin, nlZ = math.inf, u.clone(), math.inf
    with torch.no_grad():
        for i in range(1, nswap + 1):
            nlZ_t, du, dz = _vfe(hd, mean, cov, u, x, y, z)
            nlZ = float(nlZ_t)
            if nlZ < nlZmin:
                umin, nlZmin = u.clone(), nlZ
                dum, iu = du.min(0)
                dzm, iz = dz.min(0)
                if verbose:
                    print(f"{i:03d}) nlZ={nlZ:1.4e} -> {nlZ + float(dum) + float(dzm):1.4e}")
                ui = u[iu].clone()
                u[iu], z[iz] = z[iz].clone(), ui
            else:
                if verbose:
                    print(f"{i:03d}) nlZ={nlZmin:1.4e} ~< {nlZ:1.4e}")
                nlZ, u = nlZmin, umin
                break
    out = dict(hyp)
    out["xu"] = u.clone().requires_grad_(hyp["xu"].requires_grad)
    return out, nlZ
