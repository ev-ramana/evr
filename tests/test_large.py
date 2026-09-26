"""Capacity tests: there is no fixed limit on the number of variables/constraints."""

import time

import numpy as np
import pytest

from simplex_solver import LinearProgram, random_lp

try:
    from scipy.optimize import linprog
except ImportError:  # pragma: no cover - scipy is optional
    linprog = None


def scipy_objective(lp):
    if linprog is None:
        return None
    s = 1 if lp.sense == "max" else -1
    A = np.array(lp.A, dtype=float)
    b = np.array(lp.b, dtype=float)
    ub = [i for i, t in enumerate(lp.types) if t != "="]
    eq = [i for i, t in enumerate(lp.types) if t == "="]
    sg = np.array([1.0 if lp.types[i] == "<=" else -1.0 for i in ub])
    res = linprog(-s * np.array(lp.c, dtype=float),
                  A_ub=A[ub] * sg[:, None] if ub else None, b_ub=b[ub] * sg if ub else None,
                  A_eq=A[eq] if eq else None, b_eq=b[eq] if eq else None, method="highs")
    assert res.status == 0
    return -s * res.fun


def check_solution(lp, res, tol=1e-6):
    assert res.status == "optimal"
    x = np.array(res.x, dtype=float)
    lhs = np.array(lp.A, dtype=float) @ x
    b = np.array(lp.b, dtype=float)
    scale = max(1.0, float(np.max(np.abs(b))))
    for i, t in enumerate(lp.types):
        if t == "<=":
            assert lhs[i] <= b[i] + tol * scale
        elif t == ">=":
            assert lhs[i] >= b[i] - tol * scale
        else:
            assert abs(lhs[i] - b[i]) <= tol * scale
    assert np.all(x >= -tol)
    rep = res.duality()
    assert rep.strong_duality and rep.dual_feasible and rep.complementary_slackness
    ref = scipy_objective(lp)
    if ref is not None:
        assert float(res.objective) == pytest.approx(ref, rel=1e-8)


@pytest.mark.parametrize("m, n", [(100, 120), (250, 250)])
def test_large_mixed_problems(m, n):
    lp = random_lp(m, n, seed=m + n)
    start = time.perf_counter()
    res = lp.solve()
    assert time.perf_counter() - start < 60
    check_solution(lp, res)


def test_large_minimisation_uses_dual_simplex():
    rng = np.random.default_rng(7)
    m = n = 300
    lp = LinearProgram(c=rng.integers(1, 20, size=n), A=rng.integers(0, 10, size=(m, n)),
                       b=rng.integers(10, 100, size=m), types=">=", sense="min")
    res = lp.solve()
    assert res.method_key == "dual-simplex"
    check_solution(lp, res)


def test_large_degenerate_transportation():
    """Balanced transportation problems are highly degenerate."""
    rng = np.random.default_rng(11)
    S, D = 15, 20
    supply = rng.integers(20, 60, size=S)
    demand = np.full(D, supply.sum() // D)
    demand[: supply.sum() - demand.sum()] += 1
    cost = rng.integers(1, 30, size=(S, D))
    A, b, types = [], [], []
    for i in range(S):
        row = np.zeros(S * D, dtype=int)
        row[i * D:(i + 1) * D] = 1
        A.append(row); b.append(supply[i]); types.append("<=")
    for j in range(D):
        row = np.zeros(S * D, dtype=int)
        row[j::D] = 1
        A.append(row); b.append(demand[j]); types.append(">=")
    lp = LinearProgram(c=cost.ravel(), A=A, b=b, types=types, sense="min")
    for method in ("two-phase", "big-m", "dual-simplex"):
        check_solution(lp, lp.solve(method=method))


@pytest.mark.parametrize("m, n", [(5, 1500), (1500, 5)])
def test_wide_and_tall_problems(m, n):
    lp = random_lp(m, n, seed=3, mixed=False)
    check_solution(lp, lp.solve())


def test_exact_arithmetic_on_a_medium_problem():
    lp = random_lp(25, 25, seed=5)
    exact = lp.solve(exact=True)
    approx = lp.solve()
    assert exact.status == approx.status == "optimal"
    assert float(exact.objective) == pytest.approx(approx.objective, rel=1e-9)
