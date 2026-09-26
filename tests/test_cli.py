import builtins
import io
import sys

from simplex_solver.cli import main

from conftest import EXAMPLES


def run(capsys, *args):
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_full_report(capsys):
    code, out, _ = run(capsys, EXAMPLES / "wyndor.lp")
    assert code == 0
    for part in ("PROBLEM", "RESULT", "SENSITIVITY ANALYSIS", "DUALITY", "Z* = 36"):
        assert part in out


def test_steps_big_m_exact(capsys):
    code, out, _ = run(capsys, EXAMPLES / "bigm.lp", "-m", "big-m", "--steps", "--exact")
    assert code == 0 and "7M-4" in out and "STANDARD FORM" in out and "z* = 17/5" in out


def test_solve_dual_with_steps(capsys):
    code, out, _ = run(capsys, EXAMPLES / "duality.lp", "--dual", "--steps", "--exact")
    assert "DUAL PROBLEM - SIMPLEX ITERATIONS" in out
    assert "strong duality confirmed" in out


def test_options(capsys, tmp_path):
    report = tmp_path / "report.txt"
    code, out, _ = run(capsys, EXAMPLES / "diet.lp", "--no-sensitivity", "--no-duality",
                       "--digits", "2", "-o", report)
    assert code == 0 and "SENSITIVITY" not in out and "DUALITY" not in out
    assert "z* = 437.65" in out
    assert "437.65" in report.read_text()


def test_random_capacity_summary(capsys):
    code, out, _ = run(capsys, "--random", 80, 70, "--seed", 3, "--compare-scipy")
    assert code == 0 and "OPTIMAL" in out and "SENSITIVITY" not in out
    assert "scipy" in out


def test_stdin(capsys, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("max: 2x + y\nx + y <= 3\nx <= 2\n"))
    code, out, _ = run(capsys, "-")
    assert code == 0 and "z* = 5" in out


def test_errors(capsys, tmp_path):
    code, _, err = run(capsys, tmp_path / "missing.lp")
    assert code == 2 and "error" in err
    bad = tmp_path / "bad.lp"
    bad.write_text("max: x*y\nx <= 1\n")
    code, _, err = run(capsys, bad)
    assert code == 2 and "not linear" in err


def _feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(it))


def test_interactive_algebraic(capsys, monkeypatch):
    _feed(monkeypatch, ["1", "max: 3x1 + 5x2", "x1 <= 4", "2x2 <= 12", "3x1 + 2x2 <= 18", "",
                        "", "n", "y", "n", ""])
    code, out, _ = run(capsys, "--interactive")
    assert code == 0 and "z* = 36" in out and "y2 = 3/2" in out


def test_interactive_matrix(capsys, monkeypatch):
    _feed(monkeypatch, ["2", "min", "2", "2", "2 3", "1 1 >= 4", "1 3 >= 6", "", "",
                        "dual-simplex", "y", "y", "y", ""])
    code, out, _ = run(capsys, "-i")
    assert code == 0 and "z* = 9" in out and "Dual simplex" in out
