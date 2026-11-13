"""N-QPINN and A-QPINN for nonlinear integro-differential and fractional integro-PDEs.

    from qpinn import RunConfig, run_one
    run_one(RunConfig(example="6.1", method="A", seed=1234, epochs=20000, lr=1e-2))
"""
from .problems import PROBLEMS, get_problem
from .train import RunConfig, build, run_one

__all__ = ["PROBLEMS", "get_problem", "RunConfig", "build", "run_one"]
