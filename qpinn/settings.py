"""Settings used for the results of Section 6 (Table 7 of the paper).

Every run: Adam for 20 000 epochs, learning rate divided by 10 at epoch 10 000 and by 100 at epoch
16 000, then up to 500 L-BFGS iterations (strong Wolfe line search; the Adam weights are kept if
L-BFGS increases the loss). Five seeds per setting.
"""

SEEDS = [1234, 2468, 3702, 4936, 6170]

COMMON = {"epochs": 20000, "lr_steps": [[10000, 0.1], [16000, 0.01]], "lbfgs_iters": 500}

# (example, method) -> initial learning rate and QNN size (qubits x variational layers)
PAPER = {
    ("6.1", "A"): {"lr": 1e-2, "n_qubits": 8, "n_layers": 5},
    ("6.1", "N"): {"lr": 1e-2, "n_qubits": 8, "n_layers": 5},
    ("6.2", "A"): {"lr": 1e-2, "n_qubits": 6, "n_layers": 4},
    ("6.2", "N"): {"lr": 1e-2, "n_qubits": 5, "n_layers": 4},
    ("6.3", "A"): {"lr": 3e-3, "n_qubits": 6, "n_layers": 3},
    ("6.3", "N"): {"lr": 3e-3, "n_qubits": 6, "n_layers": 5},
    ("6.4", "N"): {"lr": 1e-2, "n_qubits": 4, "n_layers": 4},    # cases "smooth" and "singular"
}

METHODS = {"A": "A-QPINN", "N": "N-QPINN"}
CASES = {"6.4": ["smooth", "singular"]}
