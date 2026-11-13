"""Example 6.1: one-dimensional Volterra integro-differential equation, x in [0, 5].

    u'(x) + u(x) = lam * int_0^x e^{t-x} u(t) dt,   u(0) = 1,   lam = 1,   exact u = e^{-x} cosh x

A-QPINN: v(x) = int_0^x e^{t-x} u(t) dt  ->  u' + u - lam v = 0,  v' - u + v = 0,  u(0) = 1,  v(0) = 0
N-QPINN: 20-point Gauss-Legendre quadrature on [0, x] (Jacobian x/2 included)
"""
import numpy as np
import torch

from ..quadrature import volterra_1d
from ..utils import col, grad, mse
from .base import Problem


class Ex61Volterra1D(Problem):
    name = "6.1"
    title = "1-D Volterra IDE (Example 6.1)"
    coords, lo, hi = ["x"], [0.0], [5.0]
    fields = ["u"]
    n_out = {"A": 2, "N": 1}
    default_qnn = {"A": {"n_qubits": 8, "n_layers": 5}, "N": {"n_qubits": 8, "n_layers": 5}}
    defaults = {"n_colloc": 50, "n_val": 200, "Q": 20, "lam": 1.0}

    def sample(self, form, device):
        x = col(np.linspace(self.lo[0], self.hi[0], self.opt["n_colloc"]), device, True)
        x0 = col([self.lo[0]], device, True)
        return {"x": x, "x0": x0}

    def loss(self, net, pts, form):
        x, x0, lam = pts["x"], pts["x0"], self.opt["lam"]
        if form == "A":
            u, v = net(x).split(1, dim=1)
            r1 = grad(u, x) + u - lam * v
            r2 = grad(v, x) - u + v
            u0, v0 = net(x0).split(1, dim=1)
            c = {"res": mse(r1), "aux": mse(r2), "ic": mse(u0 - 1.0) + mse(v0)}
        else:
            u = net(x)[:, :1]
            I = volterra_1d(x, lambda t: net(t)[:, :1], lambda xx, t: torch.exp(t - xx), Q=self.opt["Q"])
            r = grad(u, x) + u - lam * I
            c = {"res": mse(r), "ic": mse(net(x0)[:, :1] - 1.0)}
        return sum(c.values()), c

    def exact(self, x):
        return {"u": torch.exp(-x) * torch.cosh(x)}

    def exact_aux(self, x):          # used by the tests
        return {"v": 0.5 * (1.0 - torch.exp(-2.0 * x))}
