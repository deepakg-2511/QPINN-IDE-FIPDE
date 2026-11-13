"""The QNN of Section 3.2.1: parameter counts of Tables 1-4 and the circuit structure of Table 5."""
import numpy as np
import pennylane as qml
import pytest
import torch

from qpinn import RunConfig, build
from qpinn.qnn import QNN, n_params, qnn_param_count
from qpinn.settings import PAPER

# trainable parameters reported in Tables 1-4
PARAMS = {("6.1", "N"): 145, ("6.1", "A"): 154, ("6.2", "N"): 82, ("6.2", "A"): 112,
          ("6.3", "N"): 115, ("6.3", "A"): 93, ("6.4", "N"): 65}


@pytest.mark.parametrize("key", sorted(PARAMS))
def test_parameter_counts_match_the_paper(key):
    ex, m = key
    ps = PAPER[key]
    _, net, size = build(RunConfig(example=ex, method=m, n_qubits=ps["n_qubits"], n_layers=ps["n_layers"]),
                         device="cpu")
    assert n_params(net) == PARAMS[key]
    assert size == {"n_qubits": ps["n_qubits"], "n_layers": ps["n_layers"]}


@pytest.mark.parametrize("d,m,nq,L", [(1, 2, 8, 5), (2, 3, 6, 3), (2, 1, 4, 4)])
def test_parameter_count_formula(d, m, nq, L):
    assert n_params(QNN(d, m, nq, L)) == qnn_param_count(d, m, nq, L)


@pytest.mark.parametrize("nq,L", [(8, 5), (5, 4), (6, 4), (6, 5), (6, 3), (4, 4)])
def test_one_cnot_per_qubit_and_layer(nq, L):
    """q*L CNOTs (Table 5); layer l uses CNOT(i, i + r_l mod q) with r_l = ((l - 1) mod (q - 1)) + 1."""
    ops = qml.StronglyEntanglingLayers(np.zeros((L, nq, 3)), wires=range(nq)).decomposition()
    cnots = [o.wires.tolist() for o in ops if o.name == "CNOT"]
    assert len(cnots) == nq * L
    for l in range(L):
        r = (l % (nq - 1)) + 1
        assert cnots[l * nq:(l + 1) * nq] == [[i, (i + r) % nq] for i in range(nq)]


def test_output_is_linear_readout_of_pauli_z():
    torch.manual_seed(0)
    q = QNN(2, 3, 4, 2)
    x = torch.rand(5, 2)
    phi = q.qlayer(q.clayer_in(x)).to(x.dtype)
    assert torch.allclose(q(x), q.clayer_out(phi), atol=1e-6)
    assert phi.abs().max().item() <= 1 + 1e-6
