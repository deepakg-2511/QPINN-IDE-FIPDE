"""Each example's loss must (nearly) vanish on the exact solution. This checks the equations, the
forcing terms, the auxiliary systems, the quadrature wiring and the initial/boundary conditions."""
import numpy as np
import pytest
import torch

from conftest import ExactNet
from qpinn.problems import PROBLEMS, get_problem

# (example, options, method, tolerance on every loss component)
CASES = [
    ("6.1", {}, "A", 1e-12), ("6.1", {}, "N", 1e-12),
    ("6.2", {}, "A", 1e-10), ("6.2", {}, "N", 1e-10),
    ("6.3", {}, "N", 1e-8),                              # 6.3 'A' has no closed-form auxiliaries
    ("6.4", {"case": "smooth"}, "N", 1e-3),              # L1 discretisation error
    ("6.4", {"case": "singular"}, "N", 1e-3),
]


@pytest.mark.parametrize("name,kw,form,tol", CASES)
def test_loss_vanishes_on_exact_solution(name, kw, form, tol):
    torch.manual_seed(0); np.random.seed(0)
    P = get_problem(name, **kw)
    _, comps = P.loss(ExactNet(P, form), P.sample(form, "cpu"), form)
    for k, v in comps.items():
        assert v.item() < tol, f"{name}/{form} {k} = {v.item():.2e}"


def test_63_forcing_term_matches_exact_solution():
    """Example 6.3 'A' residual u_x - u_t - u + g + v - u_tt = 0 with v from a fine quadrature of u."""
    from qpinn.problems.ex63_ipde_2d import g_forcing
    from qpinn.quadrature import cumtrapz_2d
    n = 401
    s = torch.linspace(0, 1, n, dtype=torch.float64)
    T, X = torch.meshgrid(s, s, indexing="ij")
    u = X + T * torch.sin(T + X)
    I = cumtrapz_2d(torch.cos(T - X) * u, s, s)
    u_t = torch.sin(T + X) + T * torch.cos(T + X)
    u_tt = 2 * torch.cos(T + X) - T * torch.sin(T + X)
    u_x = 1 + T * torch.cos(T + X)
    res = u_x - u_t - u + g_forcing(T, X) + T * I - u_tt
    assert res.abs().max().item() < 1e-4


@pytest.mark.parametrize("case", ["smooth", "singular"])
def test_64_discretisation_error_decreases(case):
    res = []
    for n_t in (16, 64):
        P = get_problem("6.4", n_t=n_t, case=case)
        res.append(P.loss(ExactNet(P, "N"), P.sample("N", "cpu"), "N")[1]["res"].item())
    assert res[1] < res[0] / 3


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_evaluate_on_exact_solution_is_zero(name):
    P = get_problem(name)
    m, arrays = P.evaluate(ExactNet(P, "N"), "cpu")
    assert m["relL2"] < 1e-6
    assert arrays["pred"].shape == arrays["exact"].shape


def test_unknown_option_rejected():
    with pytest.raises(KeyError):
        get_problem("6.1", n_points=10)


def test_64_has_only_N_and_valid_cases():
    P = get_problem("6.4")
    assert P.forms == ["N"]
    with pytest.raises(ValueError):
        P.sample("A", "cpu")
    with pytest.raises(ValueError):
        get_problem("6.4", case="other")
