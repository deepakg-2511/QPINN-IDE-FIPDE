"""Example 6.3: two-dimensional integro-partial differential equation, (t, x) in [0, 1]^2.

    u_tt = u_x - u_t - u + g(t, x) + lam * int_0^x int_0^t t cos(y1 - y2) u(y1, y2) dy1 dy2,   lam = 1
    u(0, x) = x,   u_t(0, x) = sin x,   u(t, 0) = t sin t,     exact u = x + t sin(t + x)

A-QPINN: w(t, x) = int_0^t cos(y1 - x) u(y1, x) dy1,  v(t, x) = t int_0^x w(t, y2) dy2
         residual u_x - u_t - u + g + lam v - u_tt,  v_x - t w = 0,  w_t - cos(t - x) u = 0,
         w(0, x) = 0,  v(t, 0) = 0   (three-output QNN: u, v, w); 5000 collocation points
N-QPINN: cumulative product trapezoidal rule on a uniform 41 x 41 grid, which is also the collocation grid
"""
import numpy as np
import torch

from ..quadrature import cumtrapz_2d
from ..utils import col, grad, mse
from .base import Problem


def g_forcing(t, x):
    """forcing term g(t, x) for the exact solution u = x + t sin(t + x) (checked in tests/test_problems.py)"""
    s, c = torch.sin, torch.cos
    return (t * (-t ** 2 * s(t) ** 2 / 4 + s(t) + s(t) ** 2 / 8)
            + t * (x * c(x) - s(x) ** 2 / 8 - s(x))
            - t * (-t ** 2 * s(t - x) * s(t + x) / 4 + t * x * s(t - x) * s(t + x) / 4
                   - t * x * c(t - x) * c(t + x) / 4 + x * s(t - x) * c(t + x) / 8
                   + x * s(t + x) * c(t - x) / 8 + x * c(t - x) + s(t - x) + s(t - x) * s(t + x) / 8)
            + x + s(t + x) + 2 * c(t + x) - 1)


class Ex63IPDE2D(Problem):
    name = "6.3"
    title = "2-D integro-PDE (Example 6.3)"
    coords, lo, hi = ["x", "t"], [0.0, 0.0], [1.0, 1.0]
    fields = ["u"]
    n_out = {"A": 3, "N": 1}
    default_qnn = {"A": {"n_qubits": 6, "n_layers": 3}, "N": {"n_qubits": 6, "n_layers": 5}}
    defaults = {"nf": 5000, "ni": 100, "nb": 50, "mesh": 200, "n_grid": 41, "n_val": 100,
                "lam": 1.0}

    def _ic_bc_points(self, device):
        xi = col(np.linspace(0, 1, self.opt["ni"]), device, True)
        tb = col(np.linspace(0, 1, self.opt["nb"]), device, True)
        return {"x_i": xi, "t_i": torch.zeros_like(xi).requires_grad_(True),
                "x_b": torch.zeros_like(tb).requires_grad_(True), "t_b": tb}

    def sample(self, form, device):
        pts = self._ic_bc_points(device)
        if form == "A":
            m = self.opt["mesh"]
            X, T = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m))
            idx = np.random.choice(X.size, self.opt["nf"], replace=False)
            pts.update({"x": col(X.ravel()[idx], device, True), "t": col(T.ravel()[idx], device, True)})
        else:
            n = self.opt["n_grid"]
            ts, xs = np.linspace(0, 1, n), np.linspace(0, 1, n)
            Tg, Xg = np.meshgrid(ts, xs, indexing="ij")           # axis 0: t (y1), axis 1: x (y2)
            pts.update({"ts": torch.tensor(ts, dtype=torch.float32, device=device),
                        "xs": torch.tensor(xs, dtype=torch.float32, device=device),
                        "x": col(Xg, device, True), "t": col(Tg, device, True), "shape": Tg.shape})
        return pts

    def _ic_bc(self, net, p, with_aux):
        oi = net(p["x_i"], p["t_i"])
        ob = net(p["x_b"], p["t_b"])
        ui, ub = oi[:, :1], ob[:, :1]
        ic = mse(ui - p["x_i"]) + mse(grad(ui, p["t_i"]) - torch.sin(p["x_i"]))
        bc = mse(ub - p["t_b"] * torch.sin(p["t_b"]))
        if with_aux:
            ic = ic + mse(oi[:, 2:3])          # w(0, x) = 0
            bc = bc + mse(ob[:, 1:2])          # v(t, 0) = 0
        return ic, bc

    def loss(self, net, pts, form):
        x, t, lam = pts["x"], pts["t"], self.opt["lam"]
        out = net(x, t)
        u = out[:, :1]
        u_t = grad(u, t)
        u_tt, u_x = grad(u_t, t), grad(u, x)
        g = g_forcing(t, x)
        if form == "A":
            v, w = out[:, 1:2], out[:, 2:3]
            res = u_x - u_t - u + g + lam * v - u_tt
            aux = mse(grad(v, x) - t * w) + mse(grad(w, t) - torch.cos(t - x) * u)
            ic, bc = self._ic_bc(net, pts, True)
            c = {"res": mse(res), "aux": aux, "ic": ic, "bc": bc}
        else:
            shape = pts["shape"]
            U, Tg, Xg = u.view(*shape), t.view(*shape), x.view(*shape)
            I = cumtrapz_2d(torch.cos(Tg - Xg) * U, pts["ts"], pts["xs"])
            res = u_tt - u_x + u_t + u - g - lam * (Tg * I).reshape(-1, 1)
            ic, bc = self._ic_bc(net, pts, False)
            c = {"res": mse(res), "ic": ic, "bc": bc}
        return sum(c.values()), c

    def exact(self, x, t):
        return {"u": x + t * torch.sin(t + x)}
