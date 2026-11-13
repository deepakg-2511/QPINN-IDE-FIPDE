"""The four examples of Section 6, keyed by their section number."""
from .ex61_volterra_1d import Ex61Volterra1D
from .ex62_ide_system import Ex62IDESystem
from .ex63_ipde_2d import Ex63IPDE2D
from .ex64_fractional import Ex64Fractional

PROBLEMS = {p.name: p for p in [Ex61Volterra1D, Ex62IDESystem, Ex63IPDE2D, Ex64Fractional]}


def get_problem(name, **kw):
    if name not in PROBLEMS:
        raise KeyError(f"unknown example '{name}'. choose from {sorted(PROBLEMS)}")
    return PROBLEMS[name](**kw)
