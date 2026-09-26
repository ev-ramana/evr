"""Command line interface.

    python simplex.py problem.lp                    # solve and print the full report
    python simplex.py problem.lp --steps --exact    # show every tableau with fractions
    python simplex.py problem.lp --method big-m     # choose the method
    python simplex.py problem.lp --dual             # also solve the dual problem
    python simplex.py --interactive                 # type the problem in
    python simplex.py --random 300 200              # capacity test on a random problem
"""

from __future__ import annotations

import argparse
import re
import sys
import time

from . import __version__
from .generator import random_lp
from .model import LinearProgram
from .parser import ParseError, parse_lp, read_problem
from .report import format_problem, format_solution, format_steps, full_report
from .utils import fmt_num, heading, to_fraction

EXAMPLES = """\
examples:
  python simplex.py examples/wyndor.lp
  python simplex.py examples/diet.lp --steps --exact
  python simplex.py examples/bigm.lp --method big-m --steps --exact
  python simplex.py examples/production.lp --dual
  python simplex.py --interactive
  python simplex.py --random 300 300 --seed 1 --compare-scipy

problem file format (algebraic):
  max: 3x1 + 5x2
  subject to
  labor:    x1 <= 4
  material: 2x2 <= 12
  3x1 + 2x2 <= 18
  x3 free          # optional sign declarations (default: all variables >= 0)
"""

_REL_RE = re.compile(r"(<=|>=|=<|=>|==|<|>|=|≤|≥)")


def build_arg_parser():
    p = argparse.ArgumentParser(
        prog="simplex",
        description="Simplex method solver with duality and sensitivity analysis "
                    "(shadow prices, reduced costs, ranging).",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?",
                   help="problem file: .lp/.txt (algebraic form) or .json (matrix form); "
                        "'-' reads standard input")
    p.add_argument("-m", "--method", default="auto",
                   choices=["auto", "two-phase", "big-m", "dual-simplex"],
                   help="solution method (default: auto)")
    p.add_argument("-s", "--steps", action="store_true", help="print every simplex tableau")
    p.add_argument("-e", "--exact", action="store_true",
                   help="exact fraction arithmetic (answers like 3/2 instead of 1.5)")
    p.add_argument("-d", "--dual", action="store_true",
                   help="also solve the dual problem independently (with --steps its tableaux "
                        "are printed too)")
    p.add_argument("--no-sensitivity", action="store_true", help="skip the sensitivity analysis")
    p.add_argument("--no-duality", action="store_true", help="skip the duality analysis")
    p.add_argument("--summary", action="store_true",
                   help="print only the result (useful for very large problems)")
    p.add_argument("--digits", type=int, default=4, help="decimals shown for floats (default 4)")
    p.add_argument("--bland", action="store_true",
                   help="always use Bland's rule (smallest index) for pivoting")
    p.add_argument("--max-iter", type=int, default=None, help="iteration limit")
    p.add_argument("-o", "--output", help="also save the report to this file")
    p.add_argument("-i", "--interactive", action="store_true", help="enter the problem interactively")
    p.add_argument("--random", nargs=2, type=int, metavar=("M", "N"),
                   help="solve a random feasible problem with M constraints and N variables")
    p.add_argument("--seed", type=int, default=None, help="random seed for --random")
    p.add_argument("--compare-scipy", action="store_true",
                   help="compare the objective value with scipy.optimize.linprog (if installed)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


# --------------------------------------------------------------------------- interactive
def _ask(prompt, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    return ans if ans else (default if default is not None else "")


def _ask_yes(prompt, default=True):
    ans = _ask(prompt + " (y/n)", "y" if default else "n").lower()
    return ans.startswith("y")


def _ask_int(prompt, minimum):
    while True:
        ans = _ask(prompt)
        try:
            v = int(ans)
            if v >= minimum:
                return v
        except ValueError:
            pass
        print(f"  please enter a whole number >= {minimum}")


def _ask_numbers(prompt, count):
    while True:
        ans = _ask(prompt).replace(",", " ").split()
        try:
            nums = [to_fraction(a) for a in ans]
        except (ValueError, ZeroDivisionError):
            print("  could not read the numbers, try again (fractions like 1/2 are fine)")
            continue
        if len(nums) == count:
            return nums
        print(f"  expected {count} numbers, got {len(nums)}")


def _ask_constraint(i, n):
    while True:
        line = _ask(f"Constraint {i}: {n} coefficient(s), relation, right-hand side  (e.g. 1 2 <= 10)")
        parts = _REL_RE.split(line)
        if len(parts) != 3:
            print("  use exactly one of <=, >= or =")
            continue
        try:
            coefs = [to_fraction(a) for a in parts[0].replace(",", " ").split()]
            rhs = to_fraction(parts[2].strip())
        except (ValueError, ZeroDivisionError):
            print("  could not read the numbers, try again")
            continue
        if len(coefs) != n:
            print(f"  expected {n} coefficients, got {len(coefs)}")
            continue
        return coefs, parts[1], rhs


def _read_algebraic():
    print("Type the problem (objective first, then the constraints). "
          "Finish with an empty line or 'end'.")
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip().lower() == "end" or (not line.strip() and lines):
            break
        if line.strip():
            lines.append(line)
    return parse_lp("\n".join(lines))


def _read_matrix():
    sense = _ask("Maximize or minimize? (max/min)", "max").lower()
    sense = "min" if sense.startswith("min") else "max"
    n = _ask_int("Number of decision variables", 1)
    m = _ask_int("Number of constraints", 0)
    names = [f"x{j + 1}" for j in range(n)]
    c = _ask_numbers(f"Objective coefficients of {', '.join(names) if n <= 8 else 'x1 .. x' + str(n)}", n)
    A, b, types = [], [], []
    for i in range(m):
        coefs, rel, rhs = _ask_constraint(i + 1, n)
        A.append(coefs)
        types.append(rel)
        b.append(rhs)
    signs = [">=0"] * n
    free = _ask("Free (unrestricted) variables, e.g. 'x3 x4' (Enter: none)", "").replace(",", " ").split()
    nonpos = _ask("Non-positive variables (<= 0), e.g. 'x2' (Enter: none)", "").replace(",", " ").split()
    for group, sign in ((free, "free"), (nonpos, "<=0")):
        for v in group:
            if v not in names:
                raise ValueError(f"unknown variable {v!r}")
            signs[names.index(v)] = sign
    return LinearProgram(c=c, A=A if A else [], b=b, types=types, sense=sense, var_names=names,
                         var_signs=signs)


def interactive(args):
    print(heading("SIMPLEX SOLVER - interactive mode"))
    print("How do you want to enter the problem?")
    print("  1) algebraic form, e.g.  max: 3x1 + 5x2   then   x1 + x2 <= 4 ...")
    print("  2) step by step (number of variables, coefficients, ...)")
    while True:
        choice = _ask("Choice", "1")
        try:
            lp = _read_algebraic() if choice.strip() != "2" else _read_matrix()
            break
        except (ParseError, ValueError) as exc:
            print(f"  error: {exc}\n  let's try again.")
    print()
    print(format_problem(lp))
    print()
    small = lp.num_vars <= 12 and lp.num_constraints <= 12
    method = _ask("Method (auto / two-phase / big-m / dual-simplex)", args.method)
    if method not in ("auto", "two-phase", "big-m", "dual-simplex"):
        print("  unknown method, using auto")
        method = "auto"
    args.method = method
    args.steps = _ask_yes("Show every simplex tableau?", small)
    args.exact = _ask_yes("Use exact fractions?", small)
    args.dual = _ask_yes("Also solve the dual problem separately?", True)
    out = _ask("Save the report to a file (Enter: don't save)", "")
    args.output = out or args.output
    print()
    return lp


# --------------------------------------------------------------------------- main
def _scipy_objective(lp):
    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError:
        return None, "scipy is not installed"
    s = 1 if lp.sense == "max" else -1
    A = np.array(lp.A, dtype=float)
    b = np.array(lp.b, dtype=float)
    ub = [i for i, t in enumerate(lp.types) if t != "="]
    eq = [i for i, t in enumerate(lp.types) if t == "="]
    sg = np.array([1.0 if lp.types[i] == "<=" else -1.0 for i in ub])
    bounds = [(0, None) if v == ">=0" else (None, 0) if v == "<=0" else (None, None)
              for v in lp.var_signs]
    t0 = time.perf_counter()
    res = linprog(-s * np.array(lp.c, dtype=float),
                  A_ub=A[ub] * sg[:, None] if ub else None, b_ub=b[ub] * sg if ub else None,
                  A_eq=A[eq] if eq else None, b_eq=b[eq] if eq else None,
                  bounds=bounds, method="highs")
    elapsed = time.perf_counter() - t0
    if res.status != 0:
        return None, f"scipy status: {res.message}"
    return -s * res.fun + float(lp.objective_constant), f"{elapsed:.4f} s"


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.random:
            m, n = args.random
            lp = random_lp(m, n, seed=args.seed)
            if not args.summary and (m > 60 or n > 60):
                args.summary = True
        elif args.interactive or (args.file is None and sys.stdin.isatty()):
            lp = interactive(args)
        elif args.file in (None, "-"):
            lp = parse_lp(sys.stdin.read())
        else:
            lp = read_problem(args.file)
    except (OSError, ParseError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

    if args.exact and lp.num_vars * max(lp.num_constraints, 1) > 20000:
        print("note: exact fractions are slow for large problems; consider dropping --exact",
              file=sys.stderr)
    res = lp.solve(method=args.method, exact=args.exact, record_steps=args.steps,
                   bland=args.bland, max_iter=args.max_iter)

    if args.summary:
        parts = [heading("PROBLEM"), format_problem(lp, args.digits), "",
                 heading("RESULT"), format_solution(res, args.digits)]
        text = "\n".join(parts)
    else:
        dual_report = None
        if not args.no_duality:
            dual_report = res.duality(solve_dual=args.dual, method=args.method,
                                      record_steps=args.steps, bland=args.bland,
                                      max_iter=args.max_iter)
        text = full_report(res, steps=args.steps, sensitivity=not args.no_sensitivity,
                           duality=not args.no_duality, dual_report=dual_report,
                           digits=args.digits)
        if dual_report is not None and dual_report.dual_result is not None and args.steps:
            text += "\n" + "\n".join([heading("DUAL PROBLEM - SIMPLEX ITERATIONS"),
                                      format_steps(dual_report.dual_result, args.digits).lstrip("\n"),
                                      ""])
    if args.compare_scipy:
        obj, info = _scipy_objective(lp)
        if obj is None:
            text += f"\nscipy comparison: {info}\n"
        else:
            diff = abs(float(res.objective) - obj) if res.objective is not None else float("nan")
            text += (f"\nscipy (HiGHS) objective: {fmt_num(obj, 6)}  ({info}); "
                     f"difference: {diff:.2e}\n")
    print(text)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"(report saved to {args.output})")
        except OSError as exc:
            print(f"error: could not write {args.output}: {exc}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
