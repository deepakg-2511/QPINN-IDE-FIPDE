import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float32)


class ExactNet(torch.nn.Module):
    """Stands in for a trained network: returns the exact solution (and exact auxiliaries for 'A')."""

    def __init__(self, problem, form):
        super().__init__()
        self.P, self.form = problem, form

    def forward(self, *cols):
        ex = self.P.exact(*cols)
        outs = [ex[f] for f in self.P.fields]
        if self.form == "A":
            outs += list(self.P.exact_aux(*cols).values())
        return torch.cat(outs, dim=1)


# small problem settings for the smoke tests (fast on a CPU)
TINY = {
    "6.1": {"n_colloc": 12, "Q": 8, "n_val": 30},
    "6.2": {"n_colloc": 12, "Q": 8, "n_val": 30},
    "6.3": {"nf": 40, "ni": 8, "nb": 8, "mesh": 20, "n_grid": 7, "n_val": 12},
    "6.4": {"n_x": 7, "n_t": 6, "n_val": 12},
}


@pytest.fixture
def tiny():
    return TINY
