"""Example 6.2: system of nonlinear Volterra integro-differential equations, x in [0, 1].

    u'' = 1 - x^3/3 - (1/2)(v')^2 + mu1 int_0^x (u^2 + v^2) dt
    v'' = -1 + x^2 - x u + mu2 int_0^x (u^2 - v^2) dt
    u(0) = 1, u'(0) = 2, v(0) = -1, v'(0) = 0,   mu1 = 1/2, mu2 = 1/4,   exact u = x + e^x, v = x - e^x

A-QPINN: w1 = int_0^x (u^2 + v^2) dt, w2 = int_0^x (u^2 - v^2) dt,
         w1' = u^2 + v^2, w2' = u^2 - v^2, w1(0) = w2(0) = 0   (four-output QNN: u, v, w1, w2)
N-QPINN: 20-point Gauss-Legendre quadrature on [0, x] for both integrals
"""
import numpy as np
import torch

from ..quadrature import volterra_1d
from ..utils import col, grad, mse
from .base import Problem


class Ex62IDESystem(Problem):
    name = "6.2"
    title = "Nonlinear Volterra IDE system (Example 6.2)"
    coords, lo, hi = ["x"], [0.0], [1.0]
    fields = ["u", "v"]
    n_out = {"A": 4, "N": 2}
    default_qnn = {"A": {"n_qubits": 6, "n_layers": 4}, "N": {"n_qubits": 5, "n_layers": 4}}
    defaults = {"n_colloc": 50, "n_val": 200, "mu1": 0.5, "mu2": 0.25, "Q": 20}

    def sample(self, form, device):
        x = col(np.linspace(self.lo[0], self.hi[0], self.opt["n_colloc"]), device, True)
        return {"x": x, "x0": col([0.0], device, True)}

    def _ic(self, out0, x0):
        u0, v0 = out0[:, :1], out0[:, 1:2]
        return (mse(u0 - 1.0) + mse(grad(u0, x0) - 2.0) + mse(v0 + 1.0) + mse(grad(v0, x0)))

    def loss(self, net, pts, form):
        x, x0, mu1, mu2 = pts["x"], pts["x0"], self.opt["mu1"], self.opt["mu2"]
        out = net(x)
        u, v = out[:, :1], out[:, 1:2]
        ux, vx = grad(u, x), grad(v, x)
        uxx, vxx = grad(ux, x), grad(vx, x)
        out0 = net(x0)
        if form == "A":
            w1, w2 = out[:, 2:3], out[:, 3:4]
            r1 = 1 - x ** 3 / 3 - 0.5 * vx ** 2 + mu1 * w1 - uxx
            r2 = -1 + x ** 2 - x * u + mu2 * w2 - vxx
            a1 = u ** 2 + v ** 2 - grad(w1, x)
            a2 = u ** 2 - v ** 2 - grad(w2, x)
            c = {"res": mse(r1) + mse(r2), "aux": mse(a1) + mse(a2),
                 "ic": self._ic(out0, x0) + mse(out0[:, 2:3]) + mse(out0[:, 3:4])}
        else:
            def integrands(t):
                o = net(t)
                return torch.cat([o[:, :1] ** 2 + o[:, 1:2] ** 2, o[:, :1] ** 2 - o[:, 1:2] ** 2], dim=1)
            I = volterra_1d(x, integrands, lambda xx, t: torch.ones_like(t), Q=self.opt["Q"])
            r1 = 1 - x ** 3 / 3 - 0.5 * vx ** 2 + mu1 * I[:, :1] - uxx
            r2 = -1 + x ** 2 - x * u + mu2 * I[:, 1:2] - vxx
            c = {"res": mse(r1) + mse(r2), "ic": self._ic(out0, x0)}
        return sum(c.values()), c

    def exact(self, x):
        return {"u": x + torch.exp(x), "v": x - torch.exp(x)}

    def exact_aux(self, x):          # used by the tests
        return {"w1": 2 * x ** 3 / 3 + torch.exp(2 * x) - 1, "w2": 4 * ((x - 1) * torch.exp(x) + 1)}
