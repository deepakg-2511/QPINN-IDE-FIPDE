"""Every quadrature rule against closed forms and its order of convergence."""
import math

import numpy as np
import pytest
import torch

from qpinn.quadrature import (caputo_l1_matrix, cumtrapz_2d, gauss_legendre, graded_mesh,
                              rl_integral_matrix, volterra_1d)


def test_gauss_legendre_exact_for_polynomials():
    xi, w = gauss_legendre(10, dtype=torch.float64)
    assert abs(w.sum().item() - 2.0) < 1e-12
    for k in range(20):                                   # exact up to degree 2Q - 1
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        assert abs((w * xi ** k).sum().item() - exact) < 1e-10


def test_volterra_1d_against_closed_form():
    x = torch.linspace(0, 5, 41).reshape(-1, 1)
    I = volterra_1d(x, lambda t: torch.cos(t), lambda xx, t: torch.exp(t - xx), Q=20)
    exact = 0.5 * (torch.cos(x) + torch.sin(x)) - 0.5 * torch.exp(-x)   # int_0^x e^{t-x} cos t dt
    assert torch.max(torch.abs(I - exact)).item() < 1e-5


def test_cumtrapz_2d_bilinear_exact_and_converges():
    s, t = torch.linspace(0, 0.5, 11), torch.linspace(0, 1, 21)
    S, T = torch.meshgrid(s, t, indexing="ij")
    assert torch.allclose(cumtrapz_2d(torch.ones_like(S), s, t), S * T, atol=1e-6)
    assert torch.allclose(cumtrapz_2d(S * T, s, t), S ** 2 * T ** 2 / 4, atol=1e-6)   # bilinear: exact
    errs = []
    for n in (11, 21, 41):
        s, t = torch.linspace(0, 1, n, dtype=torch.float64), torch.linspace(0, 1, n, dtype=torch.float64)
        S, T = torch.meshgrid(s, t, indexing="ij")
        errs.append((cumtrapz_2d(torch.cos(S - T), s, t) - (1 - torch.cos(S) - torch.cos(T) + torch.cos(S - T))).abs().max().item())
    assert errs[1] < errs[0] / 3.5 and errs[2] < errs[1] / 3.5             # second order


def test_graded_mesh():
    t = graded_mesh(1.0, 8, 2.0)
    assert t[0] == 0 and t[-1] == 1 and np.all(np.diff(t) > 0)
    assert np.allclose(t, (np.arange(9) / 8) ** 2)


@pytest.mark.parametrize("alpha", [0.2, 0.4, 0.7])
def test_caputo_l1_exact_for_linear_and_converges(alpha):
    t = graded_mesh(1.0, 32, 2.0)
    C = caputo_l1_matrix(t, alpha)
    assert np.allclose(C @ t, t ** (1 - alpha) / math.gamma(2 - alpha), atol=1e-12)   # D^a t
    p, errs = 2.7, []
    for N in (32, 64, 128):
        t = graded_mesh(1.0, N, 1.0)
        exact = math.gamma(p + 1) / math.gamma(p + 1 - alpha) * t ** (p - alpha)
        errs.append(np.abs(caputo_l1_matrix(t, alpha) @ t ** p - exact).max())
    rate = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
    assert min(rate) > 2 - alpha - 0.15                                    # L1 order 2 - alpha


@pytest.mark.parametrize("beta", [0.2, 0.5])
def test_rl_integral_exact_for_linear_and_converges(beta):
    t = graded_mesh(1.0, 32, 2.0)
    Q = rl_integral_matrix(t, beta)
    assert np.allclose(Q @ t, t ** (1 + beta) / math.gamma(2 + beta), atol=1e-12)    # I^b t
    p, errs = 2.5, []
    for N in (32, 64, 128):
        t = graded_mesh(1.0, N, 1.0)
        exact = math.gamma(p + 1) / math.gamma(p + 1 + beta) * t ** (p + beta)
        errs.append(np.abs(rl_integral_matrix(t, beta) @ t ** p - exact).max())
    assert errs[2] < errs[1] < errs[0] and np.log2(errs[1] / errs[2]) > 1.8
