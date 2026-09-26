"""The :class:`LinearProgram` model and the construction of its dual problem."""

from __future__ import annotations

import numpy as np

from .utils import fmt_linear, fmt_num, to_fraction

SENSES = {
    "max": "max", "maximize": "max", "maximise": "max", "maximum": "max",
    "min": "min", "minimize": "min", "minimise": "min", "minimum": "min",
}

CONSTRAINT_TYPES = {
    "<=": "<=", "=<": "<=", "<": "<=", "≤": "<=", "le": "<=", "l": "<=",
    ">=": ">=", "=>": ">=", ">": ">=", "≥": ">=", "ge": ">=", "g": ">=",
    "=": "=", "==": "=", "eq": "=", "e": "=",
}

VAR_SIGNS = {
    ">=0": ">=0", ">= 0": ">=0", "nonneg": ">=0", "nonnegative": ">=0", "+": ">=0", "pos": ">=0",
    "<=0": "<=0", "<= 0": "<=0", "nonpos": "<=0", "nonpositive": "<=0", "-": "<=0", "neg": "<=0",
    "free": "free", "urs": "free", "unrestricted": "free", "unbounded": "free",
}

FLIP = {"<=": ">=", ">=": "<=", "=": "="}


def _numeric_array(values, what: str) -> np.ndarray:
    """Convert user input to a numpy array (float/int, or object array of Fractions)."""
    try:
        arr = np.asarray(values)
    except ValueError as exc:  # ragged nested lists
        raise ValueError(f"{what}: rows must all have the same length") from exc
    if arr.dtype.kind == "b":
        arr = arr.astype(np.int64)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{what}: all coefficients must be finite numbers")
        return arr.astype(float)
    if arr.dtype.kind in "OUS":
        if arr.size == 0:
            return arr.astype(object)
        try:
            return np.vectorize(to_fraction, otypes=[object])(arr)
        except (ValueError, TypeError, ZeroDivisionError) as exc:
            raise ValueError(f"{what}: could not read the numbers ({exc})") from exc
    raise ValueError(f"{what}: unsupported data type {arr.dtype}")


def exact_array(arr: np.ndarray) -> np.ndarray:
    """Return an object array of Fractions (exact arithmetic)."""
    if arr.size == 0:
        return np.array(arr, dtype=object)
    return np.vectorize(to_fraction, otypes=[object])(arr)


def float_array(arr: np.ndarray) -> np.ndarray:
    return np.array(arr, dtype=float)


class LinearProgram:
    """A linear program

        max / min   z = c^T x + c0
        subject to  A x  (<=, >=, =)  b      (one relation per row)
                    each x_j  >= 0,  <= 0  or  free (unrestricted)

    Parameters
    ----------
    c : objective coefficients (length n)
    A : constraint matrix (m rows, n columns)
    b : right-hand sides (length m)
    types : relation of every constraint: ``"<="``, ``">="`` or ``"="``
        (default: all ``"<="``)
    sense : ``"max"`` or ``"min"``
    var_names, con_names : optional names (default ``x1..xn`` and ``C1..Cm``)
    var_signs : sign restriction of every variable: ``">=0"`` (default),
        ``"<=0"`` or ``"free"``
    objective_constant : constant term c0 added to the objective
    objective_name : symbol used for the objective when printing (``z``)

    Numbers may be ints, floats, :class:`fractions.Fraction` or strings such as
    ``"3/4"``.  There is no fixed limit on the number of variables or
    constraints: the size is only limited by the available memory.
    """

    def __init__(self, c, A, b, types=None, sense="max", var_names=None, con_names=None,
                 var_signs=None, objective_constant=0, objective_name="z", name=None):
        key = str(sense).strip().lower()
        if key not in SENSES:
            raise ValueError(f"sense must be 'max' or 'min', got {sense!r}")
        self.sense = SENSES[key]

        self.c = _numeric_array(c, "objective coefficients c").reshape(-1)
        n = self.c.size

        A_arr = _numeric_array(A if A is not None else [], "constraint matrix A")
        if A_arr.ndim == 1:
            # [] means "no constraints"; a flat list is a single constraint
            A_arr = A_arr.reshape(0, n) if A_arr.size == 0 else A_arr.reshape(1, -1)
        if A_arr.ndim != 2 or A_arr.shape[1] != n:
            raise ValueError(f"constraint matrix A must have {n} columns (one per variable), "
                             f"got shape {A_arr.shape}")
        self.A = A_arr
        m = A_arr.shape[0]

        self.b = _numeric_array(b if b is not None else [], "right-hand side b").reshape(-1)
        if self.b.size != m:
            raise ValueError(f"right-hand side b must have {m} entries, got {self.b.size}")

        if types is None:
            types = ["<="] * m
        if isinstance(types, str):
            types = [types] * m
        if len(types) != m:
            raise ValueError(f"expected {m} constraint types, got {len(types)}")
        norm_types = []
        for t in types:
            k = str(t).strip().lower()
            if k not in CONSTRAINT_TYPES:
                raise ValueError(f"unknown constraint type {t!r} (use '<=', '>=' or '=')")
            norm_types.append(CONSTRAINT_TYPES[k])
        self.types = norm_types

        if var_signs is None:
            var_signs = [">=0"] * n
        if isinstance(var_signs, str):
            var_signs = [var_signs] * n
        if len(var_signs) != n:
            raise ValueError(f"expected {n} variable signs, got {len(var_signs)}")
        norm_signs = []
        for v in var_signs:
            k = str(v).strip().lower()
            if k not in VAR_SIGNS:
                raise ValueError(f"unknown variable sign {v!r} (use '>=0', '<=0' or 'free')")
            norm_signs.append(VAR_SIGNS[k])
        self.var_signs = norm_signs

        self.var_names = self._names(var_names, n, "x", "variable")
        self.con_names = self._names(con_names, m, "C", "constraint")
        self.objective_constant = objective_constant if objective_constant is not None else 0
        self.objective_name = objective_name or "z"
        self.name = name

    @staticmethod
    def _names(names, count, prefix, what):
        if names is None:
            return [f"{prefix}{i + 1}" for i in range(count)]
        names = [str(s).strip() for s in names]
        if len(names) != count:
            raise ValueError(f"expected {count} {what} names, got {len(names)}")
        if any(not s for s in names):
            raise ValueError(f"{what} names must not be empty")
        if len(set(names)) != len(names):
            raise ValueError(f"{what} names must be unique")
        return names

    # ------------------------------------------------------------------ sizes
    @property
    def num_vars(self) -> int:
        return self.c.size

    @property
    def num_constraints(self) -> int:
        return self.b.size

    def __repr__(self):
        label = f" {self.name!r}" if self.name else ""
        return (f"<LinearProgram{label}: {self.sense}, {self.num_vars} variables, "
                f"{self.num_constraints} constraints>")

    # ------------------------------------------------------------ duality
    def dual(self, var_prefix: str | None = None) -> "LinearProgram":
        """Build the dual problem.

        Rules (primal max  <->  dual min; read the table in either direction):

        =====================  =====================
        primal (max)           dual (min)
        =====================  =====================
        constraint i  <=       y_i >= 0
        constraint i  >=       y_i <= 0
        constraint i  =        y_i free
        variable x_j >= 0      dual constraint j  >=
        variable x_j <= 0      dual constraint j  <=
        variable x_j free      dual constraint j  =
        =====================  =====================

        The dual's variable y_i is the shadow price of primal constraint i and
        the dual's constraint j belongs to primal variable x_j.  Taking the
        dual twice gives back the original problem.
        """
        if var_prefix is None:
            var_prefix = "x" if all(v.lower().startswith("y") for v in self.var_names) else "y"
        m = self.num_constraints
        if self.sense == "max":
            sign_of = {"<=": ">=0", ">=": "<=0", "=": "free"}
            type_of = {">=0": ">=", "<=0": "<=", "free": "="}
            dual_sense = "min"
        else:
            sign_of = {">=": ">=0", "<=": "<=0", "=": "free"}
            type_of = {">=0": "<=", "<=0": ">=", "free": "="}
            dual_sense = "max"
        return LinearProgram(
            c=self.b.copy(),
            A=self.A.T.copy(),
            b=self.c.copy(),
            types=[type_of[s] for s in self.var_signs],
            sense=dual_sense,
            var_names=[f"{var_prefix}{i + 1}" for i in range(m)],
            con_names=list(self.var_names),
            var_signs=[sign_of[t] for t in self.types],
            objective_constant=self.objective_constant,
            objective_name="w" if self.objective_name != "w" else "z",
            name=f"dual of {self.name}" if self.name else "dual",
        )

    # ------------------------------------------------------------ solving
    def solve(self, method: str = "auto", exact: bool = False, record_steps: bool = False, **options):
        """Solve the problem with the simplex method.  See :func:`simplex_solver.solve`."""
        from .engine import solve
        return solve(self, method=method, exact=exact, record_steps=record_steps, **options)

    # ------------------------------------------------------------ I/O
    @classmethod
    def from_string(cls, text: str) -> "LinearProgram":
        """Parse a problem written in algebraic form (see :mod:`simplex_solver.parser`)."""
        from .parser import parse_lp
        return parse_lp(text)

    @classmethod
    def from_file(cls, path) -> "LinearProgram":
        """Read a problem from a ``.lp``/``.txt`` (algebraic) or ``.json`` (matrix) file."""
        from .parser import read_problem
        return read_problem(path)

    @classmethod
    def from_dict(cls, data: dict) -> "LinearProgram":
        from .parser import lp_from_dict
        return lp_from_dict(data)

    def to_dict(self) -> dict:
        def plain(x):
            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                return float(x)
            if hasattr(x, "denominator") and not isinstance(x, int):
                return int(x) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
            return x
        return {
            "sense": self.sense,
            "c": [plain(v) for v in self.c.tolist()],
            "A": [[plain(v) for v in row] for row in self.A.tolist()],
            "b": [plain(v) for v in self.b.tolist()],
            "types": list(self.types),
            "var_names": list(self.var_names),
            "con_names": list(self.con_names),
            "var_signs": list(self.var_signs),
            "objective_constant": plain(self.objective_constant),
        }

    # ------------------------------------------------------------ printing
    def to_text(self, digits: int = 6) -> str:
        """Algebraic text that :meth:`from_string` can read back."""
        return self.formulation(digits=digits, parsable=True)

    def formulation(self, digits: int = 4, parsable: bool = False) -> str:
        """Human readable statement of the problem."""
        head = "Maximize" if self.sense == "max" else "Minimize"
        obj = fmt_linear(self.c, self.var_names, digits, self.objective_constant)
        lines = [f"{self.sense}: {obj}"] if parsable else [f"{head}  {self.objective_name} = {obj}"]
        lines.append("subject to")
        width = max((len(n) for n in self.con_names), default=0)
        for i in range(self.num_constraints):
            lhs = fmt_linear(self.A[i], self.var_names, digits)
            rhs = fmt_num(self.b[i], digits)
            lines.append(f"  {self.con_names[i] + ':':<{width + 1}}  {lhs} {self.types[i]} {rhs}")
        lines.extend("  " + s for s in self.sign_lines(parsable))
        return "\n".join(lines)

    def sign_lines(self, parsable: bool = False):
        groups = {">=0": [], "<=0": [], "free": []}
        for name, sgn in zip(self.var_names, self.var_signs):
            groups[sgn].append(name)
        out = []
        if groups[">=0"]:
            out.append(", ".join(groups[">=0"]) + " >= 0")
        if groups["<=0"]:
            out.append(", ".join(groups["<=0"]) + " <= 0")
        if groups["free"]:
            out.append(", ".join(groups["free"]) + (" free" if parsable else " free (unrestricted in sign)"))
        return out

    def __str__(self):
        return self.formulation()
