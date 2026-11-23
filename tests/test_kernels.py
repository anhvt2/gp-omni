import itertools

import torch

import gp_omni  # noqa: F401
from gp_omni.kernels import AdditiveKernel


def brute_force(x1, x2, ell, scales):
    ki = torch.exp(-0.5 * ((x1[:, None, :] - x2[None, :, :]) / ell) ** 2)
    d = x1.shape[-1]
    out = 0
    for r, s in enumerate(scales, start=1):
        e = sum(torch.prod(ki[..., list(c)], -1) for c in itertools.combinations(range(d), r))
        out = out + s * e
    return out


def test_additive_matches_brute_force_elementary_polynomials():
    torch.manual_seed(0)
    x1, x2 = torch.rand(7, 5), torch.rand(4, 5)
    k = AdditiveKernel(5, max_degree=3)
    k.initialize(lengthscale=0.7)
    k.order_scale = [0.5, 1.5, 2.0]
    got = k(x1, x2).to_dense()
    torch.testing.assert_close(got, brute_force(x1, x2, 0.7, [0.5, 1.5, 2.0]))


def test_additive_is_psd_and_diag_consistent():
    torch.manual_seed(1)
    x = torch.rand(30, 6)
    k = AdditiveKernel(6, max_degree=2, ard=True)
    K = k(x).to_dense()
    assert torch.linalg.eigvalsh(K).min() > -1e-9
    torch.testing.assert_close(k(x, diag=True), K.diagonal())


def test_order_one_is_sum_of_1d_kernels():
    torch.manual_seed(2)
    x = torch.rand(9, 4)
    k = AdditiveKernel(4, max_degree=1)
    k.initialize(lengthscale=0.4)
    k.order_scale = [1.0]
    expected = sum(torch.exp(-0.5 * ((x[:, None, i] - x[None, :, i]) / 0.4) ** 2) for i in range(4))
    torch.testing.assert_close(k(x).to_dense(), expected)
