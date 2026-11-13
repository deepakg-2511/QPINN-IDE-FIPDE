# Quantum Physics-Informed Neural Networks for Solving Integro and Fractional PDEs

Code for the numerical experiments of

> D. Gupta and R. Behera, *Quantum Physics-Informed Neural Networks for Solving Integro and Fractional
> PDEs*, submitted to SIAM Journal on Scientific Computing (2026).

It implements the two QPINN formulations of the paper:

- **N-QPINN**: the integral and fractional operators are evaluated by numerical quadrature, and local
  derivatives by automatic differentiation of the quantum trial solution (Section 5.1).
- **A-QPINN**: the integral terms are replaced by auxiliary functions, so that the integro-differential
  equation becomes a coupled local system solved by one multi-output QNN (Section 5.2).

All circuits are simulated exactly with PennyLane (`default.qubit`) and trained with PyTorch.

## Contents

| Paper | Problem | File | Methods |
|---|---|---|---|
| Example 6.1, Table 1 | 1-D Volterra IDE on [0, 5] | `qpinn/problems/ex61_volterra_1d.py` | N-QPINN, A-QPINN |
| Example 6.2, Table 2 | system of nonlinear Volterra IDEs on [0, 1] | `qpinn/problems/ex62_ide_system.py` | N-QPINN, A-QPINN |
| Example 6.3, Table 3 | 2-D integro-PDE on [0, 1]² | `qpinn/problems/ex63_ipde_2d.py` | N-QPINN, A-QPINN |
| Example 6.4, Table 4 | time-fractional integro-PDE on [0, 1]², smooth and weakly singular cases | `qpinn/problems/ex64_fractional.py` | N-QPINN |

```
qpinn/
  qnn.py           the QNN trial function (Section 3.2.1) and input scaling to [-1, 1]
  quadrature.py    Gauss-Legendre Volterra rule, cumulative 2-D trapezoidal rule, graded mesh,
                   L1 scheme (Caputo derivative), product trapezoidal rule (Riemann-Liouville integral)
  problems/        the four examples: equations, exact solutions, training points and losses
  settings.py      the settings of Table 7
  train.py         training (Adam + L-BFGS), checkpointing, saving of results
  plotting.py      training curves and solution / error plots for every run
run.py             command line: reproduces any setting of Tables 1-4
summarize.py       mean +- standard deviation over seeds
tests/             exact-solution checks of every loss, quadrature convergence orders, parameter counts,
                   short training runs
```

## Installation

```bash
git clone https://github.com/deepakg-2511/QPINN-IDE-FIPDE.git && cd QPINN-IDE-FIPDE
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 56 tests, about 20 s on a CPU
```

The results of the paper were obtained with Python 3.12, PyTorch 2.6.0 and PennyLane 0.44.1 (pinned in
`requirements.txt`). The tests check, among other things, that each loss vanishes on the exact solution
and that the parameter counts of Tables 1–4 are reproduced.

## Reproducing the results

Each command trains the five seeds of the paper (1234, 2468, 3702, 4936, 6170) with the settings of
Table 7. A GPU is used automatically if one is available (`--device cpu` forces the CPU).

```bash
python run.py --example 6.1 --method N       # Table 1, N-QPINN
python run.py --example 6.1 --method A       # Table 1, A-QPINN
python run.py --example 6.2                  # Table 2, both methods
python run.py --example 6.3                  # Table 3, both methods
python run.py --example 6.4                  # Table 4, smooth and weakly singular cases
python run.py --all                          # everything above

python summarize.py                          # mean +- std of the relative L2 error per setting
```

Useful options:

```bash
python run.py --all --dry-run                                   # list the runs, QNN sizes and parameter counts
python run.py --example 6.4 --case singular --seeds 1234        # one case, one seed
python run.py --example 6.1 --method A --seeds 1 --epochs 50 --lbfgs 0 --out scratch   # quick check
python run.py --example 6.2 --method A --n-qubits 4 --n-layers 3                       # another QNN size
python run.py --example 6.3 --method A --problem-kw nf=2000                            # problem options
```

An interrupted run is resumed by running the same command again (a checkpoint is written every 500
epochs). Finished runs are skipped. Every run writes to `results/<example>/<method>[_<case>]/seed_<seed>/`:

| File | Content |
|---|---|
| `config.json` | full configuration, QNN size, problem options, number of parameters |
| `history.csv` | loss components every epoch, validation errors every 100 epochs |
| `metrics.json` | final and best relative L2 error (per field for Example 6.2), seconds per epoch |
| `model.pt` | trained QNN parameters |
| `predictions.npz` | validation points, prediction and exact solution |
| `curves.png`, `solution.png` | training curves; solution and pointwise error |

The relative L2 error is computed on a uniform validation grid (200 points in 1-D, 100 × 100 in 2-D).

## Settings (Table 7)

| | Ex. 6.1 | Ex. 6.2 | Ex. 6.3 | Ex. 6.4 |
|---|---|---|---|---|
| Domain | [0, 5] | [0, 1] | [0, 1]² | [0, 1]² |
| Collocation points | 50 | 50 | 5000 (A), 41 × 41 (N) | 33 × 64, graded in t |
| Initial / boundary points | 1 / – | 1 / – | 100 / 50 | 33 / 2 × 65 |
| Quadrature (N-QPINN) | Gauss–Legendre, 20 nodes | Gauss–Legendre, 20 nodes | cumulative trapezoidal, 41 × 41 | L1 + product trapezoidal, ρ = 2 |
| QNN, N-QPINN (qubits × layers) | 8 × 5 | 5 × 4 | 6 × 5 | 4 × 4 |
| QNN, A-QPINN (qubits × layers) | 8 × 5 | 6 × 4 | 6 × 3 | – |
| Initial learning rate | 10⁻² | 10⁻² | 3 × 10⁻³ | 10⁻² |

All runs use Adam for 20 000 epochs, with the learning rate divided by 10 after 10 000 and by 100 after
16 000 epochs, followed by up to 500 L-BFGS iterations with a strong Wolfe line search; the Adam weights are
kept if L-BFGS increases the loss. Inputs are rescaled to [−1, 1].

## The QNN

`qpinn/qnn.py` implements the trial function of Section 3.2.1:

1. an affine map z = W x + b from the d inputs to one angle per qubit;
2. angle encoding, one single-qubit rotation per qubit (`qml.AngleEmbedding`, rotation `X`);
3. L variational layers (`qml.StronglyEntanglingLayers`): a general rotation RZ·RY·RZ on every qubit,
   followed by a ring of CNOT gates, CNOT(i, i + r mod n), with range r = ((l − 1) mod (n − 1)) + 1 in
   layer l, so n CNOTs per layer;
4. the expectation values ⟨Z₁⟩, …, ⟨Zₙ⟩ and a linear readout to the solution fields (and, for
   A-QPINN, the auxiliary functions).

The number of trainable parameters is d·n + n + 3·L·n + n·m + m for n qubits, L layers and m outputs.

## Computing time

Training time per run, Adam phase, measured on an NVIDIA RTX 4500 Ada GPU (Table 5 of the paper):

| | 6.1 N | 6.1 A | 6.2 N | 6.2 A | 6.3 N | 6.3 A | 6.4 N |
|---|---|---|---|---|---|---|---|
| s / 100 epochs | 21.2 | 11.0 | 12.2 | 15.1 | 27.7 | 62.3 | 5.4 |
| hours / run | 1.2 | 0.6 | 0.7 | 0.8 | 1.5 | 3.5 | 0.3 |

Independent runs can be started in parallel, e.g. one process per example or per seed (`--seeds`).

## Notes

- **Classical baselines.** The A-PINN and fPINN columns of Tables 1–4 were obtained with the same
  pipeline (loss, training points, quadrature, optimizer and schedule), with the QNN replaced by a tanh
  MLP of comparable parameter count; the A-PINN baselines of Examples 6.1 and 6.2 use an initial
  learning rate of 10⁻³. This repository contains the quantum methods only.
- **Reproducibility.** With the same hardware and software versions, a seed gives the same result.
  Across devices (CPU vs GPU) or library versions, floating-point differences change individual runs;
  the mean and spread over the five seeds are the quantities reported in the paper.

## Citation

```bibtex
@article{GuptaBehera2026QPINN,
  author  = {Gupta, Deepak and Behera, Ratikanta},
  title   = {Quantum Physics-Informed Neural Networks for Solving Integro and Fractional {PDEs}},
  journal = {submitted to SIAM Journal on Scientific Computing},
  year    = {2026}
}
```

## License

MIT; see `LICENSE`.
