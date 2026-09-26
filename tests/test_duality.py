from fractions import Fraction as F

import pytest

from simplex_solver import LinearProgram, parse_lp, random_lp, read_problem

from conftest import EXAMPLES


def test_dual_construction_max():
    lp = parse_lp("""
        max: 5x1 + 12x2 + 4x3
        x1 + 2x2 + x3 <= 10
        2x1 - x2 + 3x3 = 8
        x1 + x3 >= 1
        x2 free
        x3 <= 0
    """)
    d = lp.dual()
    assert d.sense == "min"
    assert list(d.c) == [10, 8, 1]
    assert [list(r) for r in d.A] == [[1, 2, 1], [2, -1, 0], [1, 3, 1]]
    assert list(d.b) == [5, 12, 4]
    assert d.var_signs == [">=0", "free", "<=0"]       # <=, =, >=  constraints
    assert d.types == [">=", "=", "<="]                # x1 >= 0, x2 free, x3 <= 0
    assert d.var_names == ["y1", "y2", "y3"] and d.con_names == ["x1", "x2", "x3"]


def test_dual_construction_min():
    lp = parse_lp("min: 2x1 + 3x2\nx1 + x2 >= 4\nx1 - x2 <= 1\nx1 + 2x2 = 5\nx2 <= 0")
    d = lp.dual()
    assert d.sense == "max"
    assert d.var_signs == [">=0", "<=0", "free"]
    assert d.types == ["<=", ">="]


def test_dual_of_dual_is_primal():
    lp = read_problem(EXAMPLES / "duality.lp")
    dd = lp.dual().dual()
    assert dd.sense == lp.sense and dd.types == lp.types and dd.var_signs == lp.var_signs
    assert dd.to_dict()["A"] == lp.to_dict()["A"]
    assert list(dd.b) == list(lp.b) and list(dd.c) == list(lp.c)
    assert dd.var_names == lp.var_names


def test_taha_duality_example():
    res = read_problem(EXAMPLES / "duality.lp").solve(exact=True)
    assert res.shadow_prices == [F(29, 5), F(-2, 5)]
    assert res.dual_solution == {"y1": F(29, 5), "y2": F(-2, 5)}
    rep = res.duality(solve_dual=True)
    assert rep.dual_objective == rep.primal_objective == F(274, 5)
    assert rep.strong_duality and rep.dual_feasible and rep.complementary_slackness
    assert rep.dual_result.status == "optimal"
    assert rep.dual_result.x == [F(29, 5), F(-2, 5)]
    assert rep.recovered_x == res.x                    # primal read from the dual tableau


@pytest.mark.parametrize("seed", range(30))
def test_strong_duality_random(seed):
    lp = random_lp(5 + seed % 4, 4 + seed % 5, seed=seed, sense="min" if seed % 3 else "max")
    res = lp.solve(exact=True)
    rep = res.duality(solve_dual=True)
    assert res.status == "optimal"
    assert rep.strong_duality and rep.dual_feasible and rep.complementary_slackness
    assert rep.dual_result.objective == res.objective
    # the dual's shadow prices form an optimal primal solution
    x = rep.recovered_x
    obj = sum(F(int(c)) * v for c, v in zip(lp.c, x))
    assert obj == res.objective


def test_unbounded_primal_has_infeasible_dual():
    res = read_problem(EXAMPLES / "unbounded.lp").solve()
    rep = res.duality(solve_dual=True)
    assert res.status == "unbounded"
    assert rep.dual_result.status == "infeasible"
    assert "infeasible" in rep.conclusion


def test_infeasible_primal():
    res = read_problem(EXAMPLES / "infeasible.lp").solve()
    rep = res.duality(solve_dual=True)
    assert rep.dual_result.status == "unbounded"
    assert rep.conclusion == "The primal is infeasible and the dual is unbounded."
    both = parse_lp("max: x1\nx1 - x2 <= -1\n-x1 + x2 <= -1")   # classic: both infeasible
    rep = both.solve().duality(solve_dual=True)
    assert rep.dual_result.status == "infeasible"


def test_duality_report_text(wyndor):
    text = wyndor.solve(exact=True).report(solve_dual=True)
    assert "Minimize  w = 4y1 + 12y2 + 18y3" in text
    assert "y1 = 0, y2 = 3/2, y3 = 1" in text
    assert "Complementary slackness is satisfied" in text
    assert "strong duality confirmed" in text


def test_shadow_prices_match_dual_solution_for_all_signs():
    lp = LinearProgram(c=[3, -1, 2], A=[[1, 1, 1], [2, -1, 0], [0, 1, 3]], b=[10, 4, -3],
                       types=["<=", ">=", "="], sense="max", var_signs=[">=0", "free", "<=0"])
    res = lp.solve(exact=True)
    dres = lp.dual().solve(exact=True)
    assert res.status == dres.status == "optimal"
    assert dres.objective == res.objective
    assert res.duality().dual_feasible


def test_problem_without_constraints_has_dual_without_variables():
    res = parse_lp("min: x + 2y\nx, y >= 0").solve(exact=True)
    rep = res.duality(solve_dual=True)
    assert rep.dual.num_vars == 0
    assert rep.dual_result.status == "optimal" and rep.dual_result.objective == 0
    rep = parse_lp("max: x\nx >= 0").solve().duality(solve_dual=True)
    assert rep.dual_result.status == "infeasible"


def test_big_m_shadow_prices_on_degenerate_problem():
    """Artificial variables left basic at zero level must be pivoted out, otherwise
    the shadow prices read from the Big-M tableau would contain M terms."""
    lp = LinearProgram.from_file(EXAMPLES / "transportation.json")
    ref = lp.solve(method="two-phase", exact=True)
    res = lp.solve(method="big-m", exact=True)
    assert res.objective == ref.objective == 1020
    rep = res.duality()
    assert rep.dual_feasible and rep.complementary_slackness
