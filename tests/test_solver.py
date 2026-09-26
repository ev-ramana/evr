from fractions import Fraction as F

import numpy as np
import pytest

from simplex_solver import LinearProgram, parse_lp, read_problem, solve
from simplex_solver.engine import Tableau, build_standard_form

from conftest import EXAMPLES, METHODS

# (file, objective, solution) - all well known textbook answers
KNOWN = [
    ("wyndor.lp", 36, [2, 6]),
    ("reddy_mikks.lp", 21, [3, F(3, 2)]),
    ("diet.lp", F(7440, 17), [F(8000, 17), F(5600, 17)]),
    ("bigm.lp", F(17, 5), [F(2, 5), F(9, 5)]),
    ("duality.lp", F(274, 5), [F(26, 5), F(12, 5), 0]),
    ("free_variables.lp", 20, [5, -1, -2]),
    ("degenerate.lp", F(5, 4), [1, 0, 1, 0]),
    ("transportation.json", 1020, None),
    ("matrix_form.json", 36, [2, 6]),
]


@pytest.mark.parametrize("method", METHODS + ["auto"])
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("fname, objective, x", KNOWN)
def test_textbook_examples(fname, objective, x, method, exact):
    lp = read_problem(EXAMPLES / fname)
    res = lp.solve(method=method, exact=exact)
    assert res.status == "optimal"
    if exact:
        assert res.objective == objective
        if x is not None:
            assert res.x == x
        assert all(isinstance(v, F) for v in res.x)
    else:
        assert res.objective == pytest.approx(float(objective), abs=1e-9)
        if x is not None:
            assert res.x == pytest.approx([float(v) for v in x], abs=1e-9)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("exact", [True, False])
def test_infeasible(method, exact):
    res = read_problem(EXAMPLES / "infeasible.lp").solve(method=method, exact=exact)
    assert res.status == "infeasible"
    assert res.objective is None and res.x is None
    assert "infeasible" in res.message


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("exact", [True, False])
def test_unbounded(method, exact):
    res = read_problem(EXAMPLES / "unbounded.lp").solve(method=method, exact=exact)
    assert res.status == "unbounded"
    u = res.unbounded
    assert u["rate"] > 0
    # every point of the ray is feasible and the objective grows
    lp = res.lp
    x0, d = np.array(u["x0"], dtype=float), np.array(u["direction"], dtype=float)
    for t in (0, 10, 1000):
        x = x0 + t * d
        assert np.all(np.array(lp.A, dtype=float) @ x <= np.array(lp.b, dtype=float) + 1e-9)
        assert np.all(x >= -1e-9)


def test_alternative_optima():
    res = read_problem(EXAMPLES / "alternative_optima.lp").solve(exact=True)
    assert res.objective == 10
    alt = res.alternative
    assert alt is not None and alt["x"] is not None
    assert alt["x"] != res.x
    assert 2 * alt["x"][0] + 4 * alt["x"][1] == 10          # also optimal
    assert sorted([tuple(res.x), tuple(alt["x"])]) == [(0, F(5, 2)), (3, 1)]


def test_alternative_optima_along_a_ray():
    lp = parse_lp("max: x1 - x2\nx1 - x2 <= 1")          # optimal along x1 - x2 = 1
    res = lp.solve(exact=True)
    assert res.objective == 1 and res.alternative["ray"]


def test_unique_optimum_has_no_alternative(wyndor):
    assert wyndor.solve().alternative is None


def test_redundant_equations():
    lp = parse_lp("min: 2x + 3y\nx + y = 4\n2x + 2y = 8\nx >= 1")
    for method in METHODS:
        res = lp.solve(method=method, exact=True)
        assert res.objective == 8 and res.x == [4, 0]
    res = lp.solve(method="two-phase", exact=True)
    assert res.redundant


def test_negative_rhs_and_zero_rhs():
    a = parse_lp("max: x1 + x2\n-x1 - x2 >= -4\nx1 - x2 >= 0").solve(exact=True)
    b = parse_lp("max: x1 + x2\nx1 + x2 <= 4\n-x1 + x2 <= 0").solve(exact=True)
    assert a.objective == b.objective == 4
    assert a.method == "Simplex method"          # no artificial variables were needed


def test_problems_without_constraints():
    assert parse_lp("max: x\nx >= 0").solve().status == "unbounded"
    res = parse_lp("min: x + 2y\nx, y >= 0").solve(exact=True)
    assert res.status == "optimal" and res.objective == 0 and res.x == [0, 0]
    res = LinearProgram(c=[-1, 2], A=[], b=[], sense="max").solve()
    assert res.status == "unbounded"


def test_objective_constant():
    res = parse_lp("max: 3x + 5\nx <= 2").solve(exact=True)
    assert res.objective == 11
    assert res.duality().dual_objective == 11


def test_cycling_example_terminates_with_blands_rule():
    """Beale's example cycles under Dantzig's rule with the 'topmost row' tie-break.
    The solver must detect the degenerate run and switch to Bland's rule."""
    lp = read_problem(EXAMPLES / "degenerate.lp")
    sf = build_standard_form(lp, "two-phase", True)
    tab = Tableau(sf, cycle_guard=12)
    original = tab._ratio_test

    def topmost_row(k, bland=False):
        if bland:
            return original(k, bland)
        col = tab.T[: tab.m, k]
        rows = np.flatnonzero(col > 0)
        if rows.size == 0:
            return None
        ratios = tab.T[rows, tab.N] / col[rows]
        return int(rows[np.argmin(ratios)])

    tab._ratio_test = topmost_row
    assert tab.run_primal("z", "Simplex") == "optimal"
    assert tab.bland_used
    assert tab.value("z") == F(5, 4)


@pytest.mark.parametrize("pricing", ["dantzig", "steepest-edge"])
def test_pricing_rules_and_bland(wyndor, pricing):
    assert wyndor.solve(pricing=pricing).objective == pytest.approx(36)
    assert wyndor.solve(pricing=pricing, bland=True, exact=True).objective == 36


def test_iteration_limit(wyndor):
    res = wyndor.solve(max_iter=1)
    assert res.status == "iteration_limit"


def test_recorded_steps(wyndor):
    res = wyndor.solve(record_steps=True, exact=True)
    assert len(res.steps) == res.iterations + 1
    assert res.steps[0].kind == "pivot" and res.steps[-1].kind == "optimal"
    text = res.report()
    assert "Tableau 0 (initial)" in text and "Entering variable: x2" in text


def test_big_m_tableau_is_symbolic():
    res = read_problem(EXAMPLES / "bigm.lp").solve(method="big-m", exact=True, record_steps=True)
    text = res.report()
    assert "7M-4" in text and "9M" in text and "(5/3)M+1/3" in text


def test_two_phase_steps_show_both_phases():
    res = read_problem(EXAMPLES / "bigm.lp").solve(method="two-phase", exact=True,
                                                  record_steps=True)
    assert set(res.phase_iterations) == {"Phase 1", "Phase 2"}
    assert "end of Phase 1" in res.report()


def test_dual_simplex_is_chosen_automatically():
    res = read_problem(EXAMPLES / "diet.lp").solve()
    assert res.method_key == "dual-simplex"
    res = read_problem(EXAMPLES / "wyndor.lp").solve()
    assert res.method == "Simplex method"


def test_dual_simplex_with_cost_shifting():
    # a max problem with >= rows is not dual feasible at the start
    lp = parse_lp("max: 2x + 3y\nx + y >= 2\nx + 2y <= 6\nx <= 4")
    res = lp.solve(method="dual-simplex", exact=True)
    assert res.status == "optimal" and res.objective == 11
    assert "cost shifting" in res.method


def test_solve_function_and_bad_method(wyndor):
    assert solve(wyndor).objective == pytest.approx(36)
    with pytest.raises(ValueError):
        wyndor.solve(method="magic")
    with pytest.raises(ValueError):
        wyndor.solve(pricing="random")


def test_numpy_and_fraction_inputs():
    lp = LinearProgram(c=np.array([3.0, 5.0]), A=np.array([[1, 0], [0, 2], [3, 2]]),
                       b=np.array([4, 12, 18]))
    assert lp.solve(exact=True).objective == 36
    lp = LinearProgram(c=[F(1, 3), F(1, 6)], A=[[1, 1]], b=[F(3, 2)])
    assert lp.solve(exact=True).objective == F(1, 2)


def test_badly_scaled_costs():
    # Phase 1 must not stop early just because the costs are large numbers
    lp = parse_lp("max: 1000000x1 + 2000000x2\nx1 + x2 >= 0.000001\n0.5x1 + x2 <= 3\n"
                  "x1 - x2 = 0.0000005")
    for method in METHODS:
        assert lp.solve(method=method).objective == pytest.approx(6e6)
        assert lp.solve(method=method, exact=True).objective == 6000000
