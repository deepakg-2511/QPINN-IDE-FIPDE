"""Common interface of the four examples of Section 6.

A problem knows its domain, its exact solution, how to sample the training points for each
formulation ('A' = A-QPINN, auxiliary functions; 'N' = N-QPINN, numerical quadrature) and how to build
the loss. The QNN only sees coordinates rescaled to [-1, 1] (see qnn.PhysicsNet).
"""
import numpy as np
import torch

from ..utils import col


class Problem:
    name = "base"
    title = ""
    coords = ["x"]            # input coordinate names, in the order the network receives them
    lo, hi = [0.0], [1.0]
    fields = ["u"]            # exact solution fields; they are the first outputs of the network
    n_out = {"A": 2, "N": 1}  # outputs per formulation (solution fields first, auxiliaries after)
    default_qnn = {"A": {"n_qubits": 6, "n_layers": 3}, "N": {"n_qubits": 6, "n_layers": 3}}
    defaults = {}

    def __init__(self, **kw):
        unknown = set(kw) - set(self.defaults)
        if unknown:
            raise KeyError(f"{self.name}: unknown option(s) {sorted(unknown)}; valid: {sorted(self.defaults)}")
        self.opt = {**self.defaults, **kw}

    # ---- to implement --------------------------------------------------------
    def sample(self, form, device):
        raise NotImplementedError

    def loss(self, net, pts, form):
        """returns (total_loss_tensor, {component_name: tensor})"""
        raise NotImplementedError

    def exact(self, *cols):
        """returns {field: (N,1) tensor} for every name in self.fields"""
        raise NotImplementedError

    # ---- shared ----------------------------------------------------------------
    @property
    def forms(self):
        return list(self.n_out)

    def val_points(self, device):
        """Full validation grid. Returns (list of column tensors, grid shape or None)."""
        n = self.opt.get("n_val", 200)
        if len(self.coords) == 1:
            return [col(np.linspace(self.lo[0], self.hi[0], n), device)], None
        a = np.linspace(self.lo[0], self.hi[0], n)
        b = np.linspace(self.lo[1], self.hi[1], n)
        A, B = np.meshgrid(a, b, indexing="ij")
        return [col(A, device), col(B, device)], (n, n)

    @torch.no_grad()
    def evaluate(self, net, device):
        cols, shape = self.val_points(device)
        out = net(*cols)
        ex = self.exact(*cols)
        m, num, den = {}, 0.0, 0.0
        for i, f in enumerate(self.fields):
            e = out[:, i:i + 1] - ex[f]
            m[f"relL2_{f}"] = (torch.norm(e) / torch.norm(ex[f])).item()
            m[f"max_{f}"] = e.abs().max().item()
            num += torch.norm(e).item() ** 2
            den += torch.norm(ex[f]).item() ** 2
        m["relL2"] = m[f"relL2_{self.fields[0]}"]
        if len(self.fields) > 1:
            m["relL2_combined"] = (num / den) ** 0.5
        arrays = {"coords": np.hstack([c.cpu().numpy() for c in cols]),
                  "pred": out[:, :len(self.fields)].cpu().numpy(),
                  "exact": np.hstack([ex[f].cpu().numpy() for f in self.fields]),
                  "grid_shape": np.array(shape if shape else [len(cols[0])])}
        return m, arrays
