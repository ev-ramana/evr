"""simplex_solver - a teaching-oriented linear programming solver.

Features
--------
* Simplex method: two-phase, Big-M (symbolic M) and dual simplex
* Maximisation and minimisation, ``<=``, ``>=`` and ``=`` constraints,
  non-negative, non-positive and free (unrestricted) variables
* Every tableau iteration can be recorded and printed
* Exact fraction arithmetic (``exact=True``) or fast floating point (numpy)
* Duality: construction of the dual problem, the dual solution read from the
  final tableau, strong duality and complementary slackness checks, and
  solving the dual problem independently
* Sensitivity analysis: shadow prices (worth of each constraint), reduced
  costs, ranges of optimality (objective coefficients) and ranges of
  feasibility (right-hand sides)
* No fixed size limit - problems with hundreds of constraints and variables
  are solved in seconds

Quick start
-----------
>>> from simplex_solver import LinearProgram
>>> lp = LinearProgram.from_string('''
... max: 3x1 + 5x2
... subject to
...   x1 <= 4
...   2x2 <= 12
...   3x1 + 2x2 <= 18
... ''')
>>> res = lp.solve()
>>> res.objective, res.x, res.shadow_prices
(36.0, [2.0, 6.0], [0.0, 1.5, 1.0])
"""

from .analysis import ConstraintResult, DualityReport, SolveResult, VariableResult, duality_analysis
from .engine import solve
from .generator import random_lp
from .model import LinearProgram
from .parser import ParseError, lp_from_dict, parse_lp, read_problem
from .report import full_report

__version__ = "1.0.0"

__all__ = [
    "LinearProgram", "solve", "parse_lp", "read_problem", "lp_from_dict", "ParseError",
    "SolveResult", "VariableResult", "ConstraintResult", "DualityReport", "duality_analysis",
    "full_report", "random_lp", "__version__",
]
