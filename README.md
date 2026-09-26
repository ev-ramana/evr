# Simplex Solver with Duality and Sensitivity Analysis

A linear programming solver written in Python for operations research coursework.
It solves problems with the **simplex method** and prints every tableau when asked.
It also reports everything you usually compute by hand from the optimal tableau:

* **Duality**: builds the dual problem, reads the optimal dual solution from the final
  tableau, checks strong duality and complementary slackness, and can solve the
  dual independently with the simplex method.
* **Worth of constraints (shadow prices / dual prices)**: how much the optimal objective
  changes per extra unit of each right-hand side (resource).
* **Reduced costs** of all variables.
* **Sensitivity analysis (ranging)**: range of optimality for every objective coefficient
  and range of feasibility for every right-hand side (allowable increase and decrease).
* **Capacity**: there is no fixed limit on the number of variables or constraints. Problems
  with hundreds of constraints and variables solve in seconds (see [Capacity](#capacity)).

```
python simplex.py examples/wyndor.lp --steps --exact --dual
```

---

## Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Writing a problem](#writing-a-problem)
5. [Command-line options](#command-line-options)
6. [Understanding the output](#understanding-the-output)
7. [Duality](#duality)
8. [Sensitivity analysis](#sensitivity-analysis)
9. [Capacity](#capacity)
10. [Using it from Python](#using-it-from-python)
11. [How it works](#how-it-works)
12. [Project structure and tests](#project-structure-and-tests)

---

## Features

| Area | What is supported |
|---|---|
| Objective | maximise or minimise, optional constant term |
| Constraints | `<=`, `>=`, `=` and ranges such as `2 <= x1 + x2 <= 10`; negative right-hand sides are allowed |
| Variables | non-negative (default), non-positive (`x <= 0`) and free / unrestricted (`x free`) |
| Methods | simplex, **two-phase**, **Big-M** (with a symbolic `M`, shown as `7M-4`), **dual simplex** |
| Arithmetic | fast floating point (numpy) or **exact fractions** (`--exact`, answers like `3/2`) |
| Iterations | every tableau with the entering and leaving variables, ratio test and pivot element |
| Special cases | infeasible, unbounded (with the improving ray), alternative optima (with a second optimal solution), degeneracy, redundant constraints |
| Anti-cycling | automatic switch to Bland's rule after repeated degenerate pivots |
| Duality | dual formulation, dual solution, strong duality, dual feasibility, complementary slackness, solving the dual separately |
| Sensitivity | shadow prices, reduced costs, slack/surplus, binding status, allowable increase/decrease, ranges |
| Input | algebraic text (`.lp`/`.txt`), JSON (matrix form), interactive prompts, Python API |
| Size | limited only by memory; steepest-edge pricing is used automatically for large problems |

## Installation

Python 3.8+ and numpy are required.

```bash
git clone https://github.com/ev-ramana/evr.git
cd evr
pip install -r requirements.txt        # just numpy
python simplex.py --help
```

You can also install it as a package, which adds a `simplex` command:

```bash
pip install .              # or: pip install -e ".[test]"  to also get pytest + scipy
simplex examples/wyndor.lp
```

In **Google Colab / Jupyter**:

```python
!pip install git+https://github.com/ev-ramana/evr.git
from simplex_solver import LinearProgram
```

## Quick start

```bash
python simplex.py examples/wyndor.lp                      # full report
python simplex.py examples/wyndor.lp --steps --exact      # every tableau, with fractions
python simplex.py examples/bigm.lp -m big-m -s -e         # Big-M method, step by step
python simplex.py examples/bigm.lp -m two-phase -s -e     # the same problem with two-phase
python simplex.py examples/diet.lp -m dual-simplex -s     # dual simplex method
python simplex.py examples/duality.lp --dual --exact      # also solve the dual problem
python simplex.py examples/reddy_mikks.lp -o report.txt   # save the report to a file
python simplex.py --interactive                           # type a problem in
python simplex.py --random 500 500 --compare-scipy        # capacity test
```

### Examples included

| File | Shows |
|---|---|
| `examples/wyndor.lp` | classic product mix (Hillier & Lieberman), max, `<=` |
| `examples/reddy_mikks.lp` | worth of resources / shadow prices (Taha) |
| `examples/diet.lp` | minimisation with `>=`, solved by the dual simplex |
| `examples/bigm.lp` | `=`, `>=`, `<=` together, for Big-M and two-phase |
| `examples/duality.lp` | dual of a problem with an equation (free dual variable) |
| `examples/free_variables.lp` | free and non-positive variables |
| `examples/degenerate.lp` | Beale's cycling example (anti-cycling) |
| `examples/alternative_optima.lp` | multiple optimal solutions |
| `examples/infeasible.lp`, `examples/unbounded.lp` | special cases |
| `examples/transportation.json` | transportation problem in JSON (row style) |
| `examples/matrix_form.json` | JSON matrix form `c`, `A`, `b` |

## Writing a problem

### Algebraic form (`.lp` or `.txt`)

```text
# comments start with '#' or '//'
max z = 3x1 + 5x2              # or: max: ..., maximize ..., min ..., minimize ...
subject to                     # optional (also 's.t.', 'st')
  plant1:  x1        <= 4      # optional name followed by ':'
  plant2:       2x2  <= 12
          3x1 + 2x2  <= 18
  2 <= x1 + x2 <= 10           # a range becomes two constraints
  x1, x2 >= 0                  # optional: all variables are >= 0 by default
```

* Coefficients: `3x1`, `3 x1`, `3*x1`, `1/2 x1`, `(1/2)x1`, `x1/2`, `2.5e-3 x1`.
  Variables and constants may appear on both sides (`2x + 3 <= y + 10`).
* Relations: `<=`, `>=`, `=` (also `<`, `>`, `==`, `=<`, `=>`, `≤`, `≥`).
* Sign declarations: `x3 <= 0` (non-positive), `x4 free` (also `urs`, `unrestricted`),
  `x1, x2 >= 0`. A line with only variables compared to `0` is read as a **sign
  restriction**, not a constraint.
* Several statements can share a line if separated by `;`. `end` stops reading.

### JSON (matrix form), useful for large problems

```json
{
  "sense": "max",
  "c": [3, 5],
  "A": [[1, 0], [0, 2], [3, 2]],
  "b": [4, 12, 18],
  "types": ["<=", "<=", "<="],
  "var_names": ["x1", "x2"],
  "con_names": ["plant1", "plant2", "plant3"],
  "var_signs": [">=0", ">=0"]
}
```

Only `c`, `A` and `b` are required (default: `max`, all `<=`, all variables `>= 0`).
A row style with `"constraints": [{"name": ..., "coefficients": [...], "type": ">=", "rhs": 8}]`
is also accepted (see `examples/transportation.json`). Numbers may be written as
strings like `"1/3"` to keep them exact.

### Interactive

`python simplex.py -i` asks for the problem, either typed in algebraic form or step by
step (number of variables, number of constraints, coefficients, ...). There are no
limits on the counts. It then asks which method to use and whether to show the tableaux.

## Command-line options

| Option | Meaning |
|---|---|
| `-m, --method {auto,two-phase,big-m,dual-simplex}` | method (default `auto`, see below) |
| `-s, --steps` | print the standard form and every simplex tableau |
| `-e, --exact` | exact fraction arithmetic |
| `-d, --dual` | also solve the dual problem independently (with `--steps` its tableaux too) |
| `--no-sensitivity`, `--no-duality` | leave out those sections |
| `--summary` | only the result (handy for very large problems) |
| `--digits N` | decimals shown for floating point numbers (default 4) |
| `--bland` | always use Bland's rule (smallest subscript) for pivoting |
| `--max-iter N` | iteration limit |
| `-o FILE` | also save the report to a file |
| `-i, --interactive` | enter the problem interactively |
| `--random M N [--seed S]` | solve a random feasible problem with M constraints and N variables |
| `--compare-scipy` | check the optimal value against `scipy.optimize.linprog` (if installed) |

`--method auto` uses the plain simplex method when all constraints are `<=` with
`b >= 0`. It uses the dual simplex when the slack basis is dual feasible, for example a
minimisation with non-negative costs and `>=` constraints. Otherwise it uses two-phase.

## Understanding the output

The report has these sections: **PROBLEM**, then **STANDARD FORM** and **SIMPLEX
ITERATIONS** (with `--steps`), then **RESULT**, **SENSITIVITY ANALYSIS**, **DUALITY**
and **NOTES**.

Tableaux use the usual textbook layout. The objective row holds `z_j - c_j`:

* maximisation: optimal when all `z_j - c_j >= 0`; the entering variable has the most negative value
* minimisation: optimal when all `z_j - c_j <= 0`; the entering variable has the most positive value
* Phase 1 of the two-phase method minimises `r = a1 + a2 + ...`
* Big-M rows are written with a symbolic `M`, e.g. `7M-4`, `(5/3)M+1/3`

Numbers are shown as decimals rounded to `--digits` places, or as exact fractions
everywhere (problem statement and tableaux alike) with `--exact`.

Example (`python simplex.py examples/wyndor.lp --steps --exact`):

```
Tableau 0 (initial)   [z = 0]
  Basis  x1  x2  s1  s2  s3  RHS      Ratio
  -----  --  --  --  --  --  ---  ---------
  s1      1   0   1   0   0    4          -
  s2      0   2   0   1   0   12  6  <- min
  s3      3   2   0   0   1   18          9
  -----  --  --  --  --  --  ---  ---------
  z      -3  -5   0   0   0    0
Entering variable: x2  (most negative z_j - c_j = -5)
Leaving variable:  s2  (minimum ratio 6)
Pivot element: 2
...
Tableau 2 - optimal   [z = 36]
  Basis  x1  x2  s1    s2    s3  RHS
  -----  --  --  --  ----  ----  ---
  s1      0   0   1   1/3  -1/3    2
  x2      0   1   0   1/2     0    6
  x1      1   0   0  -1/3   1/3    2
  -----  --  --  --  ----  ----  ---
  z       0   0   0   3/2     1   36
Optimal: all z_j - c_j >= 0.
```

Slack variables are named `s1, s2, ...`, surplus variables `e1, ...` and artificial
variables `a1, ...`, numbered after the constraint they belong to. A free variable `x` is
split into `x+` and `x-`, and a non-positive variable becomes `x'` with `x = -x'`.

## Duality

For every problem the report shows the **dual problem** and the **optimal dual
solution** read from the final primal tableau (`y = c_B B^-1`). It then checks:

* **strong duality**: `z* = w*`
* **dual feasibility**: every dual constraint and sign restriction is satisfied
* **complementary slackness**: `y_i * slack_i = 0` and `x_j * (dual slack)_j = 0`

With `--dual` the dual problem is also **solved from scratch** by the simplex method. The
primal solution is then read back from the dual's shadow prices, as a check in both
directions. When the primal is unbounded the report explains that the dual is infeasible.
When the primal is infeasible, `--dual` shows whether the dual is unbounded or infeasible.

The dual is built with the standard rules (read the table from left to right for a max
primal, or from right to left for a min primal):

| Primal (max) | Dual (min) |
|---|---|
| constraint `i` is `<=` | `y_i >= 0` |
| constraint `i` is `>=` | `y_i <= 0` |
| constraint `i` is `=` | `y_i` free |
| `x_j >= 0` | dual constraint `j` is `>=` |
| `x_j <= 0` | dual constraint `j` is `<=` |
| `x_j` free | dual constraint `j` is `=` |
| objective coefficients `c` | right-hand sides |
| right-hand sides `b` | objective coefficients |

Taking the dual of the dual gives back the original problem.

## Sensitivity analysis

Sign conventions are the same as Excel Solver's sensitivity report:

| Quantity | Meaning |
|---|---|
| **Shadow price** `y_i` (worth of constraint `i`) | change of the optimal objective per unit increase of `b_i`; valid inside the range of feasibility. It equals the optimal value of dual variable `y_i`. It is 0 for a constraint that is not binding. |
| **Reduced cost** `d_j = c_j - sum_i y_i a_ij` | change of the objective per unit increase of `x_j` from its current value. It is 0 for basic variables, `<= 0` for non-basic variables of a max problem and `>= 0` for a min problem. It also tells you how much `c_j` must improve before `x_j` becomes positive. |
| **Range of optimality** of `c_j` | objective coefficients (one at a time) for which the current solution stays optimal |
| **Range of feasibility** of `b_i` | right-hand sides (one at a time) for which the current basis stays feasible, so the shadow prices stay valid |
| **Slack / surplus**, **binding** | unused amount of a `<=` resource / excess over a `>=` requirement; binding means slack = 0 |

Example (`examples/reddy_mikks.lp`, matches Taha's textbook):

```
  Variable  Value  Reduced cost  Obj. coef  Allow. increase  Allow. decrease  Range of optimality
  x1            3             0          5                1                3               [2, 6]
  x2          3/2             0          4                6              2/3           [10/3, 10]

  Constraint        LHS  Type  RHS  Slack/Surplus  Status       Shadow price  ...  Range of feasibility
  raw_material_M1    24  <=     24        0 slack  binding               3/4  ...  [20, 36]
  raw_material_M2     6  <=      6        0 slack  binding               1/2  ...  [4, 20/3]
  market_limit     -3/2  <=      1      5/2 slack  not binding             0  ...  [-3/2, inf]
  demand_limit      3/2  <=      2      1/2 slack  not binding             0  ...  [3/2, inf]
```

So one extra ton of raw material M1 is worth 3/4 (i.e. $750). That value holds while
the M1 supply stays between 20 and 36 tons.

## Capacity

There is **no fixed limit** on the number of constraints or variables; only memory
limits the size. The tableau engine uses
vectorised numpy operations. For large problems it switches automatically to
**steepest-edge pricing** (and dual steepest edge in the dual simplex), which needs far
fewer iterations than the textbook rule. Small problems and step-by-step output still use
the textbook rule (most negative `z_j - c_j`), so the tableaux match hand calculations.

`python benchmarks/benchmark.py --big` on random dense problems (the objective values
agree with scipy's HiGHS solver):

| Problem size (constraints x variables) | Mixed `<=`,`>=`,`=` (two-phase) | Minimisation, all `>=` (dual simplex) |
|---|---|---|
| 100 x 100 | 0.03 s | 0.01 s |
| 200 x 200 | 0.16 s | 0.03 s |
| 300 x 300 | 0.55 s | 0.06 s |
| 500 x 500 | 2.2 s | 0.18 s |
| 1000 x 1000 | ~45 s | 1.4 s |

Exact fraction arithmetic (`--exact`) is meant for small, textbook-sized problems,
because fractions get slow on large ones.

## Using it from Python

```python
from simplex_solver import LinearProgram

lp = LinearProgram.from_string("""
    max z = 3x1 + 5x2
    subject to
      plant1: x1 <= 4
      plant2: 2x2 <= 12
      plant3: 3x1 + 2x2 <= 18
""")
# or: LinearProgram(c=[3, 5], A=[[1, 0], [0, 2], [3, 2]], b=[4, 12, 18],
#                   types=["<=", "<=", "<="], sense="max")
# or: LinearProgram.from_file("examples/wyndor.lp")

res = lp.solve(method="two-phase", exact=True, record_steps=True)
res.status                     # 'optimal'
res.objective                  # Fraction(36, 1)
res.solution                   # {'x1': Fraction(2, 1), 'x2': Fraction(6, 1)}
res.dual_values                # shadow prices {'plant1': 0, 'plant2': 3/2, 'plant3': 1}
res.reduced_costs              # [0, 0]
res.variables[0].cost_range    # (0, 15/2)  range of optimality of c1
res.constraints[1].rhs_range   # (6, 18)    range of feasibility of b2
print(res.report())            # the full text report (with the tableaux)

dual = lp.dual()               # the dual problem as a LinearProgram
print(dual)                    # Minimize  w = 4y1 + 12y2 + 18y3 ...
rep = res.duality(solve_dual=True)
rep.strong_duality             # True
rep.complementary_slackness    # True
rep.dual_result.x              # [0, 3/2, 1]  (the dual solved by the simplex method)
```

Without `exact=True` the numbers are ordinary floats (`36.0`, `1.5`, ...).
`solve()` options: `method` (`auto`, `two-phase`, `big-m`, `dual-simplex`), `exact`,
`record_steps`, `bland`, `pricing` (`auto`, `dantzig`, `steepest-edge`), `tol` and
`max_iter`.

## How it works

1. **Standard form.** Every problem is converted to an internal maximisation
   `max c^T x, A x = b, x >= 0`. Variables `x <= 0` become `x = -x'`, free variables become
   `x = x+ - x-`, and rows are multiplied by -1 where needed. Then slack (`<=`),
   surplus (`>=`) and artificial variables (`>=`, `=`) are added.
2. **Simplex iterations** run on a dense tableau `[B^-1 A | B^-1 b]` with Gauss-Jordan pivots.
   * *Two-phase*: Phase 1 minimises the sum of the artificial variables. If that minimum
     is greater than 0 the problem is infeasible. Artificial variables left in the basis
     at zero level are pivoted out, or the row is reported as redundant. Phase 2 then
     optimises `z`.
   * *Big-M*: the objective row is kept as two rows, the coefficient of `M` and the
     constant part, and compared lexicographically. The result is exact without choosing
     a numeric value for `M`.
   * *Dual simplex*: the most infeasible basic variable leaves, and the entering variable
     comes from the ratio test `|z_j - c_j| / |a_rj|`. If the starting basis is not dual
     feasible, the costs are temporarily shifted to find a feasible basis, and the primal
     simplex finishes.
   * *Anti-cycling*: after 50 consecutive degenerate pivots the solver switches to
     Bland's rule until the objective improves again. Bland's rule cannot cycle, so the
     method always terminates.
3. **Reading the results** from the optimal tableau:
   * primal solution `x_B = B^-1 b`
   * dual solution / shadow prices `y = c_B B^-1`. These are the objective-row entries
     under the starting basis columns, adjusted for row signs and the max/min conversion.
   * reduced costs `d_j = c_j - y^T a_j`
   * range of optimality of `c_j`: all reduced costs stay optimal, `d + t g >= 0`, where
     `g` is the row of the basic variable in the tableau, or the unit vector for a non-basic one
   * range of feasibility of `b_i`: the basic variables stay non-negative,
     `x_B + t B^-1 e_i >= 0`
4. In floating point mode, the final tableau is recomputed from the original data
   (`B^-1 A` via an LU solve). This also happens during long runs when rounding errors
   grow, so the reported numbers are accurate.

## Project structure and tests

```
simplex.py                 run without installing:  python simplex.py problem.lp
simplex_solver/
  model.py                 LinearProgram class, construction of the dual problem
  parser.py                algebraic text and JSON input
  engine.py                standard form, tableau, primal / Big-M / dual simplex
  analysis.py              solution, shadow prices, reduced costs, ranging, duality checks
  report.py                text report (tableaux, sensitivity and duality tables)
  cli.py                   command-line interface and interactive mode
  generator.py             random feasible test problems
examples/                  textbook problems (answers in the comments of each file)
tests/                     pytest test suite
benchmarks/benchmark.py    capacity benchmark
```

Run the tests (about 220 tests, a few seconds):

```bash
pip install pytest scipy     # scipy is optional; it is used as an independent reference
python -m pytest
```

The tests check the textbook answers (Wyndor, Reddy Mikks, the diet problem, Taha's
Big-M and duality examples, Beale's cycling example) with all three methods, in both
exact and floating point arithmetic. They also check:

* the duality theorems on random problems;
* every sensitivity range, by re-solving the problem with perturbed data;
* large random problems against scipy.
