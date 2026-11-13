"""Example 6.4: nonlinear time-fractional integro-partial differential equation, (x, t) in [0, 1]^2.

    D_t^a1 u + D_t^a2 u - u_xx + I_t^b u - u^2 = f(x, t),   u(x, 0) = 0,  u(0, t) = u(1, t) = 0
    a1 = 0.7, a2 = 0.4, b = 0.2,   exact u = t^p sin(pi x)
        case "smooth":    p = 4 + a1
        case "singular":  p = a1   (d_t u unbounded at t = 0)

Only N-QPINN applies (the weakly singular kernels admit no finite auxiliary system, Remark 5.1):
Caputo derivatives by the L1 scheme, the Riemann-Liouville integral by the product trapezoidal rule,
both on the graded mesh t_n = (n/N)^rho with rho = 2; 33 x 64 space-time grid.
"""
import math

import numpy as np
import torch

from ..quadrature import caputo_l1_matrix, graded_mesh, rl_integral_matrix
from ..utils import col, grad, mse
from .base import Problem


class Ex64Fractional(Problem):
    name = "6.4"
    title = "Time-fractional integro-PDE (Example 6.4)"
    coords, lo, hi = ["x", "t"], [0.0, 0.0], [1.0, 1.0]
    fields = ["u"]
    n_out = {"N": 1}
    default_qnn = {"N": {"n_qubits": 4, "n_layers": 4}}
    defaults = {"a1": 0.7, "a2": 0.4, "beta": 0.2, "case": "smooth",
                "rho": 2.0, "n_x": 33, "n_t": 64, "n_val": 100}

    def __init__(self, **kw):
        super().__init__(**kw)
        if self.opt["case"] not in ("smooth", "singular"):
            raise ValueError(f"6.4: case must be 'smooth' or 'singular', not {self.opt['case']!r}")

    @property
    def p(self):
        return 4.0 + self.opt["a1"] if self.opt["case"] == "smooth" else self.opt["a1"]

    def f(self, x, t):
        a1, a2, b, p = self.opt["a1"], self.opt["a2"], self.opt["beta"], self.p
        G = math.gamma
        tp = lambda e: torch.pow(t, e)
        time = (G(p + 1) / G(p + 1 - a1) * tp(p - a1) + G(p + 1) / G(p + 1 - a2) * tp(p - a2)
                + math.pi ** 2 * tp(p) + G(p + 1) / G(p + 1 + b) * tp(p + b))
        s = torch.sin(math.pi * x)
        return time * s - tp(2 * p) * s ** 2

    def sample(self, form, device):
        if form != "N":
            raise ValueError("Example 6.4 has only the N formulation (no finite auxiliary system, Remark 5.1)")
        xs = np.linspace(0, 1, self.opt["n_x"])
        ts = graded_mesh(1.0, self.opt["n_t"], self.opt["rho"])
        Xg, Tg = np.meshgrid(xs, ts, indexing="ij")            # axis 0: x, axis 1: t
        to_t = lambda M: torch.tensor(M, dtype=torch.float32, device=device)
        return {"x": col(Xg, device, True), "t": col(Tg, device), "shape": Xg.shape,
                "C1": to_t(caputo_l1_matrix(ts, self.opt["a1"])),
                "C2": to_t(caputo_l1_matrix(ts, self.opt["a2"])),
                "Qb": to_t(rl_integral_matrix(ts, self.opt["beta"]))}

    def loss(self, net, pts, form):
        x, t, shape = pts["x"], pts["t"], pts["shape"]
        u = net(x, t)[:, :1]
        u_xx = grad(grad(u, x), x)
        U, Uxx = u.view(*shape), u_xx.view(*shape)
        X, T = x.view(*shape), t.view(*shape)
        R = U @ pts["C1"].T + U @ pts["C2"].T - Uxx + U @ pts["Qb"].T - U ** 2 - self.f(X, T)
        R = R[1:-1, 1:]                                        # interior x, t_n with n >= 1
        c = {"res": mse(R), "ic": mse(U[:, 0]), "bc": mse(U[0, :]) + mse(U[-1, :])}
        return sum(c.values()), c

    def exact(self, x, t):
        return {"u": torch.pow(t, self.p) * torch.sin(math.pi * x)}
