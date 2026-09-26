from fractions import Fraction as F

import numpy as np
import pytest

from simplex_solver import LinearProgram, parse_lp, random_lp, read_problem

from conftest import EXAMPLES, METHODS

INF = float("inf")


@pytest.mark.parametrize("method", METHODS)
def test_wyndor_report(wyndor, method):
    res = wyndor.solve(method=method, exact=True)
    assert res.shadow_prices == [0, F(3, 2), 1]
    assert res.reduced_costs == [0, 0]
    x1, x2 = res.variables
    assert x1.cost_range == (0, F(15, 2))
    assert x2.cost_range == (2, INF)
    p1, p2, p3 = res.constraints
    assert (p1.slack, p2.slack, p3.slack) == (2, 0, 0)
    assert not p1.binding and p2.binding and p3.binding
    assert p1.rhs_range == (2, INF)
    assert p2.rhs_range == (6, 18)
    assert p3.rhs_range == (12, 24)


def test_reddy_mikks_matches_taha(reddy_mikks):
    res = reddy_mikks.solve(exact=True)
    assert res.shadow_prices == [F(3, 4), F(1, 2), 0, 0]      # worth of M1, M2, ...
    assert res.variables[0].cost_range == (2, 6)
    assert res.variables[1].cost_range == (F(10, 3), 10)
    assert res.constraints[0].rhs_range == (20, 36)
    assert res.constraints[1].rhs_range == (4, F(20, 3))
    assert res.constraints[2].slack == F(5, 2)


def test_minimisation_signs():
    # diet problem: increasing a '>=' requirement raises the cost -> positive shadow price
    res = read_problem(EXAMPLES / "diet.lp").solve(exact=True)
    # solving y1 + 0.21 y2 = 0.3, y1 - 0.30 y2 = 0.9 by hand gives y1 = 93/170, y2 = -20/17
    assert res.shadow_prices == [F(93, 170), F(-20, 17), 0]
    # loosening the '<=' protein limit lowers the cost -> negative shadow price
    # reduced costs of non-basic variables are >= 0 in a min problem ...
    lp = parse_lp("min: 3x + 2y + 4z\nx + y + z >= 10\nx - y >= 0")
    res = lp.solve(exact=True)
    for v in res.variables:
        assert v.reduced_cost >= 0
        assert (v.reduced_cost == 0) or not v.basic
    # ... and <= 0 in a max problem
    res = read_problem(EXAMPLES / "duality.lp").solve(exact=True)
    assert res.variables[2].reduced_cost < 0 and not res.variables[2].basic


def test_reduced_cost_formula():
    """d_j = c_j - sum_i y_i a_ij"""
    for seed in range(15):
        lp = random_lp(6, 7, seed=seed, sense="max" if seed % 2 else "min")
        res = lp.solve(exact=True)
        A = lp.A
        for j, v in enumerate(res.variables):
            expect = F(int(lp.c[j])) - sum(F(int(A[i, j])) * res.shadow_prices[i] for i in range(6))
            assert v.reduced_cost == expect


def _resolve(lp, c=None, b=None):
    return LinearProgram(c=lp.c if c is None else c, A=lp.A, b=lp.b if b is None else b,
                         types=lp.types, sense=lp.sense, var_signs=lp.var_signs).solve(exact=True)


def _inside(delta):
    """A step strictly inside an allowable change (half way, or 7 if unlimited)."""
    return F(7) if delta == INF else delta / 2


@pytest.mark.parametrize("seed", range(25))
def test_ranges_by_resolving(seed):
    """Inside the range of optimality the current solution stays optimal; inside the
    range of feasibility the objective changes by exactly shadow price * change."""
    rng = np.random.default_rng(seed)
    lp = random_lp(int(rng.integers(2, 7)), int(rng.integers(2, 7)), seed=seed,
                   sense="max" if seed % 2 else "min")
    res = lp.solve(exact=True)
    assert res.status == "optimal"
    c = [F(int(v)) for v in lp.c]
    for j, v in enumerate(res.variables):
        for step in (_inside(v.allowable_increase), -_inside(v.allowable_decrease)):
            c2 = list(c)
            c2[j] += step
            new = _resolve(lp, c=c2)
            assert new.objective == sum(ci * xi for ci, xi in zip(c2, res.x))
    b = [F(int(v)) for v in lp.b]
    for i, con in enumerate(res.constraints):
        for step in (_inside(con.allowable_increase), -_inside(con.allowable_decrease)):
            if step == 0:
                continue
            b2 = list(b)
            b2[i] += step
            new = _resolve(lp, b=b2)
            assert new.status == "optimal"
            assert new.objective == res.objective + con.shadow_price * step


def test_ranges_same_for_all_methods(reddy_mikks):
    ref = reddy_mikks.solve(method="two-phase", exact=True)
    for method in ("big-m", "dual-simplex"):
        res = reddy_mikks.solve(method=method, exact=True)
        assert [v.cost_range for v in res.variables] == [v.cost_range for v in ref.variables]
        assert [c.rhs_range for c in res.constraints] == [c.rhs_range for c in ref.constraints]


def test_free_and_nonpositive_variables():
    res = read_problem(EXAMPLES / "free_variables.lp").solve(exact=True)
    assert res.shadow_prices == [2, 1, 1]
    # check the ranges by re-solving just inside each end
    lp = res.lp
    for j, v in enumerate(res.variables):
        lo, hi = v.cost_range
        for cj in (lo + F(1, 100) if lo != -INF else v.cost - 50, hi - F(1, 100) if hi != INF else v.cost + 50):
            c2 = [F(int(x)) for x in lp.c]
            c2[j] = cj
            new = _resolve(lp, c=c2)
            assert new.objective == sum(a * x for a, x in zip(c2, res.x))


def test_degenerate_solution_is_flagged():
    res = parse_lp("max: x + y\nx + y <= 2\nx <= 2\ny <= 2\nx - y <= 2").solve(exact=True)
    assert res.objective == 2 and res.x == [2, 0]
    assert res.degenerate and res.degenerate_vars == ["s2", "s4"]
    # x + y = 2 is parallel to the objective: (0, 2) is optimal as well
    assert res.alternative["x"] == [0, 2]
    assert "Degenerate solution" in res.report()


def test_report_contains_sensitivity(wyndor):
    text = wyndor.solve().report()
    assert "Shadow price" in text and "Range of optimality" in text and "Range of feasibility" in text
