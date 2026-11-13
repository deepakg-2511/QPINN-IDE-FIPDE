"""Quantum neural network (QNN) used as the trial function of N-QPINN and A-QPINN.

    x  --(affine map z = W x + b)-->  angle encoding  -->  L variational layers  -->  <Z_1>, ..., <Z_n>
       --(classical readout  u = C phi + c0)-->  [solution fields..., auxiliary fields...]

Circuit (PennyLane):
  * encoding:   qml.AngleEmbedding(z, rotation="X"), one input angle z_i per qubit
  * layers:     qml.StronglyEntanglingLayers: in layer l every qubit gets Rot = RZ RY RZ, followed by
                a ring of CNOT(i, i + r_l mod n) with range r_l = ((l - 1) mod (n - 1)) + 1
  * readout:    expectation of Pauli-Z on every qubit, then a linear layer

Trainable parameters: d*n + n (input map) + 3*L*n (rotations) + n*m + m (readout).
"""
import torch
import torch.nn as nn


class QNN(nn.Module):
    def __init__(self, d_in, n_out, n_qubits=6, n_layers=3, rotation="X"):
        super().__init__()
        import pennylane as qml

        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.clayer_in = nn.Linear(d_in, n_qubits)          # z = W x + b
        self.clayer_out = nn.Linear(n_qubits, n_out)        # u = C phi + c0
        dev = qml.device("default.qubit", wires=n_qubits)
        wires = list(range(n_qubits))

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            qml.AngleEmbedding(inputs, wires=wires, rotation=rotation)
            qml.StronglyEntanglingLayers(weights, wires=wires)
            return [qml.expval(qml.PauliZ(i)) for i in wires]

        self.qlayer = qml.qnn.TorchLayer(qnode, {"weights": (n_layers, n_qubits, 3)})

    def forward(self, x):
        return self.clayer_out(self.qlayer(self.clayer_in(x)).to(x.dtype))


class PhysicsNet(nn.Module):
    """Wraps the QNN: takes physical coordinates as separate columns and rescales them to [-1, 1]."""

    def __init__(self, model, lo, hi):
        super().__init__()
        self.model = model
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32).view(1, -1))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32).view(1, -1))

    def forward(self, *cols):
        X = torch.cat(cols, dim=1)
        return self.model(2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0)


def qnn_param_count(d, m, n_qubits, n_layers):
    return d * n_qubits + n_qubits + 3 * n_layers * n_qubits + n_qubits * m + m


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
