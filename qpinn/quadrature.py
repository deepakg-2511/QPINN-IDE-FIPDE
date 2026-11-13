"""Quadrature rules for the nonlocal terms.

volterra_1d        int_a^x K(x, t) f(t) dt              Gauss-Legendre, Jacobian (x-a)/2 included
cumtrapz_2d        int_0^{x_i} int_0^{y_j} F(s, t)      product trapezoid on a tensor grid (all nodes at once)
graded_mesh        t_n = T (n/N)^rho
caputo_l1_matrix   D^alpha u(t_n) ~ sum_k C[n, k] u(t_k)         L1 scheme, any (graded) mesh, 0 < alpha < 1
rl_integral_matrix I^beta  u(t_n) ~ sum_k Q[n, k] u(t_k)         product trapezoid (piecewise-linear u), beta > 0
"""
import math

import numpy as np
import torch


def gauss_legendre(Q: int, device=None, dtype=torch.float32):
    xi, w = np.polynomial.legendre.leggauss(Q)
    return (torch.tensor(xi, dtype=dtype, device=device).view(1, -1),
            torch.tensor(w, dtype=dtype, device=device).view(1, -1))


def volterra_1d(x, f, kernel, Q=20, a=0.0):
    """I(x) = int_a^x kernel(x, t) f(t) dt for a column x (N, 1).

    f:      callable, (M, 1) -> (M, k) model outputs (all k columns are integrated)
    kernel: callable, (x (N,1), t (N,Q)) -> (N, Q)
    returns (N, k)
    """
    xi, w = gauss_legendre(Q, x.device, x.dtype)
    half = 0.5 * (x - a)                                   # (N, 1)  -- the (x - a)/2 Jacobian
    t = half * (xi + 1.0) + a                              # (N, Q)
    ft = f(t.reshape(-1, 1))                               # (N*Q, k)
    k = ft.shape[1]
    ft = ft.reshape(x.shape[0], Q, k)
    K = kernel(x, t).unsqueeze(-1)                         # (N, Q, 1)
    return half * torch.sum(w.unsqueeze(-1) * K * ft, dim=1)


def _cumtrapz(F, h, dim):
    """Cumulative trapezoid along `dim` with node spacings h (1-D tensor, len n-1)."""
    n = F.shape[dim]
    lo = F.narrow(dim, 0, n - 1)
    hi = F.narrow(dim, 1, n - 1)
    shape = [1] * F.dim()
    shape[dim] = n - 1
    seg = 0.5 * (lo + hi) * h.view(shape)
    zero = torch.zeros_like(F.narrow(dim, 0, 1))
    return torch.cat([zero, torch.cumsum(seg, dim=dim)], dim=dim)


def cumtrapz_2d(F, s_nodes, t_nodes):
    """F[i, j] = integrand at (s_i, t_j). Returns I[i, j] = int_0^{s_i} int_0^{t_j} F ds dt.

    Product trapezoid rule, second order, exact limits at every grid node; differentiable in F.
    s_nodes, t_nodes: 1-D tensors starting at the lower limit.
    """
    hs = s_nodes[1:] - s_nodes[:-1]
    ht = t_nodes[1:] - t_nodes[:-1]
    return _cumtrapz(_cumtrapz(F, hs, 0), ht, 1)


def graded_mesh(T: float, N: int, rho: float = 1.0) -> np.ndarray:
    return T * (np.arange(N + 1) / N) ** rho


def caputo_l1_matrix(t: np.ndarray, alpha: float) -> np.ndarray:
    """L1 scheme on a non-uniform mesh t_0 = 0 < t_1 < ... < t_N (row 0 is zero)."""
    N = len(t) - 1
    C = np.zeros((N + 1, N + 1))
    g = math.gamma(2.0 - alpha)
    for n in range(1, N + 1):
        for k in range(1, n + 1):
            b = ((t[n] - t[k - 1]) ** (1 - alpha) - (t[n] - t[k]) ** (1 - alpha)) / (t[k] - t[k - 1])
            C[n, k] += b / g
            C[n, k - 1] -= b / g
    return C


def rl_integral_matrix(t: np.ndarray, beta: float) -> np.ndarray:
    """Riemann-Liouville integral of order beta with u piecewise linear on the mesh."""
    N = len(t) - 1
    Qm = np.zeros((N + 1, N + 1))
    g = math.gamma(beta)
    for n in range(1, N + 1):
        for k in range(1, n + 1):
            a, b, h = t[n] - t[k - 1], t[n] - t[k], t[k] - t[k - 1]
            J0 = (a ** beta - b ** beta) / beta
            J1 = a * J0 - (a ** (beta + 1) - b ** (beta + 1)) / (beta + 1)
            Qm[n, k] += (J1 / h) / g
            Qm[n, k - 1] += (J0 - J1 / h) / g
    return Qm
