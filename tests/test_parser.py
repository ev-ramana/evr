from fractions import Fraction as F

import pytest

from simplex_solver import LinearProgram, ParseError, lp_from_dict, parse_lp, read_problem

from conftest import EXAMPLES


@pytest.mark.parametrize("line, sense, name", [
    ("max: 3x1 + 5x2", "max", "z"),
    ("maximize z = 3x1 + 5x2", "max", "z"),
    ("Max Z = 3x1 + 5x2", "max", "Z"),
    ("MAXIMISE: profit = 3x1 + 5x2", "max", "profit"),
    ("min: 3x1 + 5x2", "min", "z"),
    ("Minimize w = 3 x1 + 5 x2", "min", "w"),
])
def test_objective_forms(line, sense, name):
    lp = parse_lp(f"{line}\nst\nx1 + x2 <= 4")
    assert lp.sense == sense
    assert lp.objective_name == name
    assert list(lp.c) == [3, 5]


def test_objective_on_next_line():
    lp = parse_lp("Maximize\n  z = 2x + 3y\nsubject to\n  x + y <= 1")
    assert lp.var_names == ["x", "y"]
    assert list(lp.c) == [2, 3]


@pytest.mark.parametrize("expr, coefs", [
    ("3x1 + 2x2", [3, 2]),
    ("3 x1 + 2 * x2", [3, 2]),
    ("1/2 x1 - (3/4)x2", [F(1, 2), F(-3, 4)]),
    ("x1/2 + 0.25x2", [F(1, 2), F(1, 4)]),
    ("-x1 - x2", [-1, -1]),
    ("2(x1 + x2) - x2", [2, 1]),
    ("2.5e-1 x1 + 1e1 x2", [F(1, 4), 10]),
])
def test_coefficient_forms(expr, coefs):
    lp = parse_lp(f"max: {expr}\nx1 + x2 <= 1")
    assert list(lp.c) == coefs


def test_constraints_are_normalised():
    lp = parse_lp("""
        max: x + y
        subject to
        c1: 2x + 3 <= y + 10      # variables and constants on both sides
        4 >= x                    # constant on the left
        x - y = -1
        1 <= x + y <= 5           # range -> two constraints
        a: 0 <= x + y <= 7        # named range
    """)
    assert lp.con_names == ["c1", "C1", "C2", "C3", "C4", "a_lo", "a_hi"]
    assert [list(r) for r in lp.A] == [[2, -1], [1, 0], [1, -1], [1, 1], [1, 1], [1, 1], [1, 1]]
    assert list(lp.b) == [7, 4, -1, 1, 5, 0, 7]
    assert lp.types == ["<=", "<=", "=", ">=", "<=", ">=", "<="]


def test_sign_declarations():
    lp = parse_lp("""
        min: x1 + x2 + x3 + x4 + x5
        x1 + x2 + x3 + x4 + x5 >= 1
        x1, x2 >= 0
        x3 <= 0
        x4 free
        0 >= x5
    """)
    assert lp.var_signs == [">=0", ">=0", "<=0", "free", "<=0"]
    lp = parse_lp("max: a + b\na + b <= 1\na >= 0, b <= 0; c urs")
    assert lp.var_signs == [">=0", "<=0", "free"]
    lp = parse_lp("max: a + b\na + b <= 1\na, b are unrestricted in sign")
    assert lp.var_signs == ["free", "free"]


def test_single_variable_constraint_is_a_constraint():
    lp = parse_lp("max: x1\nx1 <= 4\nx1 >= 1")
    assert lp.num_constraints == 2 and lp.var_signs == [">=0"]


def test_unicode_relations_and_comments():
    lp = parse_lp("max: x + y  // objective\nx + y ≤ 4  # c1\nx ≥ 1\nx − y = 0")
    assert lp.types == ["<=", ">=", "="]
    assert list(lp.A[2]) == [1, -1]


def test_objective_constant_and_end():
    lp = parse_lp("min: x + 10\nx >= 2\nend\nthis line is ignored")
    assert lp.objective_constant == 10
    assert lp.solve(exact=True).objective == 12


@pytest.mark.parametrize("text, message", [
    ("x + y <= 4", "no objective"),
    ("max: x*y\nx <= 1", "not linear"),
    ("max: x\nx + 1", "relation"),
    ("max: x\nc: x <= 1\nc: x <= 2", "duplicate"),
    ("max: x\nx <= 1 <= 2 <= 3", "too many"),
    ("max: x\n3 <= 4", "no variables"),
    ("max: x\nx + y free", "free"),
    ("max: x\nx <= 1\nmin: x", "only one objective"),
    ("max: x / (y)\nx <= 1", "divide"),
])
def test_errors(text, message):
    with pytest.raises(ParseError, match=message):
        parse_lp(text)


def test_round_trip(tmp_path):
    lp = read_problem(EXAMPLES / "free_variables.lp")
    again = parse_lp(lp.to_text())
    assert again.to_dict() == lp.to_dict()
    path = tmp_path / "p.lp"
    path.write_text(lp.to_text())
    assert read_problem(path).solve(exact=True).objective == 20


def test_json_formats(tmp_path):
    m1 = read_problem(EXAMPLES / "matrix_form.json")
    assert m1.con_names == ["plant1", "plant2", "plant3"]
    assert m1.solve().objective == 36
    t = read_problem(EXAMPLES / "transportation.json")
    assert (t.num_vars, t.num_constraints, t.sense) == (12, 7, "min")
    lp = lp_from_dict({"problem": "min: x\nx >= 3"})
    assert lp.solve().objective == 3
    lp = LinearProgram.from_dict({"sense": "max", "c": ["1/2", 1], "A": [[1, 1]], "b": [4]})
    assert lp.solve(exact=True).objective == 4


def test_model_validation():
    with pytest.raises(ValueError):
        LinearProgram(c=[1, 2], A=[[1, 2, 3]], b=[1])
    with pytest.raises(ValueError):
        LinearProgram(c=[1], A=[[1]], b=[1], types=["<>"])
    with pytest.raises(ValueError):
        LinearProgram(c=[1], A=[[1]], b=[1], sense="maybe")
    with pytest.raises(ValueError):
        LinearProgram(c=[1, 1], A=[[1, 1]], b=[1], var_names=["x", "x"])
    with pytest.raises(ValueError):
        LinearProgram(c=[1, 1], A=[[1, 1], [1]], b=[1, 2])


def test_exact_str():
    from simplex_solver.utils import exact_str
    assert exact_str(F(3, 10)) == "0.3"
    assert exact_str(F(-7, 4)) == "-1.75"
    assert exact_str(F(1, 3)) == "1/3"
    assert exact_str(F(1, 2**20)) == "0.00000095367431640625"
    assert exact_str(0.1) == "0.1" and exact_str(-2.5e-7) == "-0.00000025" and exact_str(5) == "5"


def test_to_text_is_exact_for_long_numbers():
    # numbers that the rounded display format cannot show exactly
    lp = parse_lp("max: 0.1234567x + 1/3 y + 12345678901233/7 z\n"
                  "x + y + z <= 0.00000001\nx - 1/3 y >= -2.5")
    text = lp.to_text()
    assert "~" not in text
    assert parse_lp(text).to_dict() == lp.to_dict()


def test_display_follows_the_arithmetic():
    lp = parse_lp("min: 0.3x + 1/3 y\n0.21x + 1/2 y >= 1")
    assert str(lp).splitlines()[0] == "Minimize  z = 0.3x + 1/3 y"           # as typed
    assert "0.21x + 0.5y >= 1" in lp.formulation(exact=False)                 # decimals
    assert lp.formulation(exact=False).splitlines()[0].endswith("0.3x + 0.3333y")
    assert "21/100 x + 1/2 y >= 1" in lp.formulation(exact=True)              # fractions
    float_report = lp.solve().report()
    exact_report = lp.solve(exact=True).report()
    assert "3/10" not in float_report and "21/100" not in float_report
    assert "0.21y1" in float_report                                          # dual problem
    assert "Minimize  z = 3/10 x + 1/3 y" in exact_report


def test_to_text_with_names_that_look_like_exponents():
    lp = LinearProgram(c=[3, 2], A=[[1, 1], [2, 5]], b=[4, 9], var_names=["e1", "E2"])
    again = parse_lp(lp.to_text())
    assert list(again.c) == [3, 2] and again.var_names == ["e1", "E2"]
    assert again.objective_constant == 0
