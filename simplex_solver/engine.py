"""The simplex engine: standard form, the tableau and the pivoting algorithms.

Internally every problem is written as a *maximisation* in equality form

    max  c^T x   s.t.   A x = b,  x >= 0

where the columns of ``A`` are the (sign-adjusted) decision variables, then the
slack (``s``) / surplus (``e``) variables and finally the artificial (``a``)
variables.  The objective row stores the reduced costs ``z_j - c_j`` of this
maximisation, so a tableau is optimal when every entry is ``>= 0``.
(Reports convert back to the user's max/min convention.)

Three algorithms are available:

* ``two-phase`` - Phase 1 minimises the sum of artificial variables, Phase 2
  optimises the real objective.  Without artificials this is the ordinary
  simplex method.
* ``big-m`` - artificial variables get the cost ``-M``; ``M`` is kept symbolic
  (every objective entry is ``p*M + q``) so no huge number is ever used.
* ``dual-simplex`` - starts from a dual feasible basis and restores primal
  feasibility.  If the starting basis is not dual feasible the costs are
  temporarily shifted, and the primal simplex finishes the job.

Dantzig's rule (most negative reduced cost) is used for pricing.  After a run
of degenerate pivots the solver switches to Bland's rule, which cannot cycle,
so the method always terminates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .model import FLIP, LinearProgram, exact_array, float_array
from .utils import to_fraction

PRICING_RULES = ("auto", "dantzig", "steepest-edge")

METHODS = {
    "auto": "auto",
    "two-phase": "two-phase", "twophase": "two-phase", "2-phase": "two-phase", "two_phase": "two-phase",
    "simplex": "two-phase", "primal": "two-phase", "primal-simplex": "two-phase",
    "big-m": "big-m", "bigm": "big-m", "big_m": "big-m", "m": "big-m",
    "dual-simplex": "dual-simplex", "dual": "dual-simplex", "dualsimplex": "dual-simplex",
    "dual_simplex": "dual-simplex",
}


# --------------------------------------------------------------------------- standard form
@dataclass
class StandardForm:
    """The problem in equality form together with the bookkeeping needed to map
    tableau columns and rows back to the user's variables and constraints."""

    method: str
    exact: bool
    s: int                  # +1 for a max problem, -1 for a min problem (internal = s * user)
    m: int                  # number of constraints (rows)
    n: int                  # number of user variables
    N: int                  # number of tableau columns
    A: np.ndarray           # m x N
    b: np.ndarray           # m
    c: np.ndarray           # N, internal (maximisation) costs, 0 for s/e/a columns
    c0: object              # internal objective constant
    c_art: np.ndarray       # N, -1 on artificial columns (Phase 1 / Big-M part)
    names: list             # column names
    kinds: list             # 'x' decision, 's' slack, 'e' surplus, 'a' artificial
    col_var: list           # 'x' columns: user variable index; others: constraint index
    col_mult: list          # 'x' columns: +1/-1 with x_user = sum(mult * column); others 0
    row_sign: list          # row i of the standard form = row_sign[i] * (constraint i)
    row_type: list          # relation of row i after multiplying by row_sign[i]
    identity_col: list      # column that is basic in row i of the starting tableau
    is_art: np.ndarray      # bool mask of the artificial columns
    var_cols: list          # for each user variable: [(column, mult), ...]
    notes: list             # human readable remarks about the conversion
    A_user: np.ndarray      # the user's data in the same number type (float or Fraction)
    b_user: np.ndarray
    c_user: np.ndarray
    c0_user: object

    @property
    def has_artificials(self) -> bool:
        return bool(self.is_art.any())


def build_standard_form(lp: LinearProgram, method: str, exact: bool) -> StandardForm:
    conv = exact_array if exact else float_array
    dtype = object if exact else float
    zero = Fraction(0) if exact else 0.0
    one = Fraction(1) if exact else 1.0
    A = conv(lp.A)
    b = conv(lp.b)
    c = conv(lp.c)
    s = 1 if lp.sense == "max" else -1
    c0_user = to_fraction(lp.objective_constant) if exact else float(lp.objective_constant)
    c0 = s * c0_user
    m, n = A.shape
    notes = []

    # decision-variable columns (x <= 0 -> x = -x';  x free -> x = x+ - x-)
    struct = []
    for j, (name, sign) in enumerate(zip(lp.var_names, lp.var_signs)):
        if sign == ">=0":
            struct.append((j, 1, name))
        elif sign == "<=0":
            struct.append((j, -1, name + "'"))
            notes.append(f"{name} <= 0 is replaced by {name} = -{name}' with {name}' >= 0")
        else:
            struct.append((j, 1, name + "+"))
            struct.append((j, -1, name + "-"))
            notes.append(f"{name} is free: {name} = {name}+ - {name}-  with {name}+, {name}- >= 0")

    # row signs: primal methods need b >= 0, the dual simplex needs <= rows
    row_sign, row_type = [], []
    for i in range(m):
        t = lp.types[i]
        if method == "dual-simplex":
            sg = -1 if t == ">=" else 1
        else:
            sg = -1 if (b[i] < 0 or (b[i] == 0 and t == ">=")) else 1
        row_sign.append(sg)
        row_type.append(t if sg == 1 else FLIP[t])
        if sg == -1:
            notes.append(f"constraint {lp.con_names[i]} is multiplied by -1 "
                         f"({t} becomes {FLIP[t]})")

    used = set(lp.var_names) | {name for _, _, name in struct}

    def fresh(base):
        name = base
        while name in used:
            name += "'"
        used.add(name)
        return name

    extra = []  # (row, kind)
    for i in range(m):
        if row_type[i] == "<=":
            extra.append((i, "s"))
        elif row_type[i] == ">=":
            extra.append((i, "e"))
    art_rows = [i for i in range(m) if row_type[i] in (">=", "=")]
    ns = len(struct)
    N = ns + len(extra) + len(art_rows)

    A_std = np.full((m, N), zero, dtype=dtype)
    sig = np.array(row_sign, dtype=dtype)
    js = [j for j, _, _ in struct]
    mults = np.array([mult for _, mult, _ in struct], dtype=dtype)
    if m:
        A_std[:, :ns] = A[:, js] * mults[None, :] * sig[:, None]
    b_std = b * sig if m else np.array([], dtype=dtype)
    c_std = np.full(N, zero, dtype=dtype)
    c_std[:ns] = c[js] * mults * s
    c_art = np.full(N, zero, dtype=dtype)

    names = [name for _, _, name in struct]
    kinds = ["x"] * ns
    col_var = list(js)
    col_mult = [mult for _, mult, _ in struct]
    identity_col = [None] * m
    col = ns
    for i, kind in extra:
        A_std[i, col] = one if kind == "s" else -one
        if kind == "s":
            identity_col[i] = col
        names.append(fresh(f"{kind}{i + 1}"))
        kinds.append(kind)
        col_var.append(i)
        col_mult.append(0)
        col += 1
    for i in art_rows:
        A_std[i, col] = one
        identity_col[i] = col
        c_art[col] = -one
        names.append(fresh(f"a{i + 1}"))
        kinds.append("a")
        col_var.append(i)
        col_mult.append(0)
        col += 1

    var_cols = [[] for _ in range(n)]
    for k, (j, mult, _) in enumerate(struct):
        var_cols[j].append((k, mult))

    return StandardForm(
        method=method, exact=exact, s=s, m=m, n=n, N=N, A=A_std, b=b_std, c=c_std, c0=c0,
        c_art=c_art, names=names, kinds=kinds, col_var=col_var, col_mult=col_mult,
        row_sign=row_sign, row_type=row_type, identity_col=identity_col,
        is_art=np.array([k == "a" for k in kinds], dtype=bool), var_cols=var_cols, notes=notes,
        A_user=A, b_user=b, c_user=c, c0_user=c0_user,
    )


# --------------------------------------------------------------------------- tableau
@dataclass
class Step:
    """A snapshot of the tableau, recorded when ``record_steps=True``."""

    phase: str              # e.g. "Phase 1", "Phase 2", "Big-M", "Dual simplex"
    kind: str               # 'pivot', 'optimal', 'unbounded', 'infeasible', 'drive-out'
    pricing: str            # which objective row is used: 'w', 'z', 'lex' or 'dual'
    T: np.ndarray           # copy of the full tableau (m constraint rows + z row + w row)
    basis: list
    entering: int | None = None
    leaving: int | None = None   # row index
    message: str = ""


class Tableau:
    """Dense simplex tableau.

    ``T`` has ``m + 2`` rows and ``N + 1`` columns: rows ``0..m-1`` are the
    constraints, row ``m`` is the objective row (``z_j - c_j`` of the internal
    maximisation, value in the last column), row ``m + 1`` is the Phase-1 /
    Big-M row (the coefficient of ``M``).  The last column is the right-hand side.
    """

    def __init__(self, sf: StandardForm, tol: float = 1e-9, bland: bool = False,
                 max_iter: int | None = None, record: bool = False, cycle_guard: int = 50,
                 pricing: str = "auto"):
        self.sf = sf
        self.m, self.N = sf.m, sf.N
        self.exact = sf.exact
        self.zero = Fraction(0) if sf.exact else 0.0
        self.one = Fraction(1) if sf.exact else 1.0
        dtype = object if sf.exact else float
        T = np.full((self.m + 2, self.N + 1), self.zero, dtype=dtype)
        T[: self.m, : self.N] = sf.A
        T[: self.m, self.N] = sf.b
        self.T = T
        self.basis = list(sf.identity_col)
        self.is_basic = np.zeros(self.N, dtype=bool)
        self.is_basic[self.basis] = True
        self.eligible = ~sf.is_art          # artificial variables may never (re-)enter
        self.cost_z = sf.c.copy()
        self.const_z = sf.c0
        self.cost_w = sf.c_art
        if sf.exact:
            self.tol_piv = self.tol_d = self.tol_w = self.tol_feas = 0
        else:
            cmax = float(np.max(np.abs(sf.c))) if self.N else 1.0
            bmax = float(np.max(np.abs(sf.b))) if self.m else 1.0
            self.tol_piv = tol                      # pivot elements
            self.tol_d = tol * max(1.0, cmax)       # reduced costs of the objective row
            self.tol_w = tol                        # Phase-1 / M row (costs are 0 or -1)
            self.tol_feas = tol * max(1.0, bmax)    # values of the variables
        self.force_bland = bland
        self.cycle_guard = cycle_guard
        if pricing not in PRICING_RULES:
            raise ValueError(f"unknown pricing rule {pricing!r}; use auto, dantzig or steepest-edge")
        if pricing == "auto":
            # Dantzig's rule reproduces hand calculations; steepest edge needs far
            # fewer iterations on large problems.
            large = self.m * self.N > 5000
            pricing = "steepest-edge" if (large and not record and not sf.exact) else "dantzig"
        self.pricing = pricing
        self.max_iter = max_iter if max_iter is not None else max(1000, 50 * (self.m + self.N))
        self.iterations = 0
        self.phase_iterations: dict[str, int] = {}
        self.steps = [] if record else None
        self.bland_used = False
        self.unbounded_col = None
        self.infeasible_row = None
        self._since_check = 0
        self.set_objective_rows()

    # ------------------------------------------------------------ basic operations
    def set_row(self, idx, cost, const):
        """Write ``z_j - c_j = c_B^T B^-1 a_j - c_j`` for the given costs into row ``idx``."""
        m, N = self.m, self.N
        rows = self.T[:m]
        cB = cost[self.basis] if m else np.array([], dtype=self.T.dtype)
        if m:
            self.T[idx, :N] = cB @ rows[:, :N] - cost
            self.T[idx, N] = cB @ rows[:, N] + const
            self.T[idx, self.basis] = self.zero
        else:
            self.T[idx, :N] = -cost
            self.T[idx, N] = const

    def set_objective_rows(self):
        self.set_row(self.m, self.cost_z, self.const_z)
        self.set_row(self.m + 1, self.cost_w, self.zero)

    def pivot(self, r: int, k: int):
        """Gauss-Jordan pivot on row ``r``, column ``k``."""
        T = self.T
        T[r] = T[r] / T[r, k]
        col = T[:, k].copy()
        col[r] = self.zero
        nz = np.flatnonzero(col)
        if not self.exact and nz.size > 0.3 * T.shape[0]:
            T -= np.outer(col, T[r])            # dense column: in-place update is fastest
            T[nz, k] = self.zero
        elif nz.size:
            T[nz] -= np.outer(col[nz], T[r])
            T[nz, k] = self.zero
        T[r, k] = self.one
        self.is_basic[self.basis[r]] = False
        self.is_basic[k] = True
        self.basis[r] = k
        self.iterations += 1
        if not self.exact:
            self._since_check += 1
            if self._since_check >= 100:
                self._since_check = 0
                self.refresh(only_if_inaccurate=True)

    def refresh(self, only_if_inaccurate: bool = False):
        """Float mode: recompute the tableau from the original data and the current
        basis (``B^-1 A``) to wash out accumulated rounding errors."""
        if self.exact or self.m == 0:
            return
        sf, m, N = self.sf, self.m, self.N
        B = sf.A[:, self.basis]
        if only_if_inaccurate:
            resid = B @ self.T[:m, N] - sf.b
            if np.max(np.abs(resid)) <= 1e-11 * max(1.0, float(np.max(np.abs(sf.b)))):
                return
        try:
            X = np.linalg.solve(B, np.column_stack([sf.A, sf.b]))
        except np.linalg.LinAlgError:
            return
        if not np.all(np.isfinite(X)):
            return
        self.T[:m] = X
        self.T[:m, self.basis] = np.eye(m)
        self.set_objective_rows()

    def value(self, row: str):
        return self.T[self.m if row == "z" else self.m + 1, self.N]

    # ------------------------------------------------------------ recording
    def _record(self, phase, kind, pricing, entering=None, leaving=None, message=""):
        if self.steps is not None:
            self.steps.append(Step(phase, kind, pricing, self.T.copy(), list(self.basis),
                                   entering, leaving, message))

    def _count(self, phase):
        self.phase_iterations[phase] = self.phase_iterations.get(phase, 0) + 1

    # ------------------------------------------------------------ pricing & ratio tests
    def _pick_entering(self, d, extra_mask=None, bland=False, tol=None):
        tol = self.tol_d if tol is None else tol
        cand = self.eligible & ~self.is_basic & (d < -tol)
        if extra_mask is not None:
            cand &= extra_mask
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            return None
        if bland:
            return int(idx[0])                       # smallest index (Bland's rule)
        if self.pricing == "steepest-edge" and self.m:
            cols = self.T[: self.m, idx]             # d_j^2 / (1 + ||B^-1 a_j||^2)
            weights = self.one + (np.einsum("ij,ij->j", cols, cols) if not self.exact
                                  else (cols * cols).sum(axis=0))
            return int(idx[np.argmax(d[idx] ** 2 / weights)])
        return int(idx[np.argmin(d[idx])])          # most negative (Dantzig's rule)

    def _ratio_test(self, k, bland=False):
        m, N = self.m, self.N
        col = self.T[:m, k]
        rows = np.flatnonzero(col > self.tol_piv)
        if rows.size == 0:
            return None
        rhs = self.T[rows, N]
        if not self.exact:
            rhs = np.maximum(rhs, 0.0)
        ratios = rhs / col[rows]
        best = ratios.min()
        ties = rows[ratios == best] if self.exact else rows[ratios <= best + self.tol_feas]
        if ties.size == 1:
            return int(ties[0])
        if bland:
            return int(min(ties, key=lambda i: self.basis[i]))
        return int(ties[np.argmax(col[ties])])      # largest pivot element among ties

    def is_dual_feasible(self) -> bool:
        d = self.T[self.m, : self.N]
        return not np.any(self.eligible & ~self.is_basic & (d < -self.tol_d))

    # ------------------------------------------------------------ algorithms
    def run_primal(self, pricing: str, phase: str) -> str:
        """Primal simplex iterations.

        ``pricing`` = ``'z'`` (real objective), ``'w'`` (Phase 1) or ``'lex'``
        (Big-M: the M-part first, then the constant part).
        """
        m, N = self.m, self.N
        degenerate_run = 0
        bland = self.force_bland
        m_part_done = False
        while True:
            if pricing == "lex":
                k = self._pick_entering(self.T[m + 1, :N], bland=bland, tol=self.tol_w)
                if k is None:
                    if self.value("w") < -self.tol_feas:
                        self._record(phase, "infeasible", pricing,
                                     message="No entering variable, but an artificial variable is "
                                             "still positive: the problem is infeasible.")
                        return "infeasible"
                    if not m_part_done:
                        # The M part is optimal and every artificial is 0.  Artificials that
                        # are still basic (at zero level) are pivoted out, exactly as after
                        # Phase 1; otherwise the shadow prices would contain M terms.
                        m_part_done = True
                        self.drive_out_artificials(phase, pricing="lex")
                    w = self.T[m + 1, :N]
                    flat = (w == 0) if self.exact else (np.abs(w) <= self.tol_w)
                    k = self._pick_entering(self.T[m, :N], extra_mask=flat, bland=bland)
            elif pricing == "z":
                k = self._pick_entering(self.T[m, :N], bland=bland)
            else:
                k = self._pick_entering(self.T[m + 1, :N], bland=bland, tol=self.tol_w)
            if k is None:
                self._record(phase, "optimal", pricing)
                return "optimal"
            r = self._ratio_test(k, bland)
            if r is None:
                self.unbounded_col = k
                self._record(phase, "unbounded", pricing, entering=k)
                return "unbounded"
            step = self.T[r, N] / self.T[r, k]
            msg = "Bland's rule (anti-cycling) is active." if bland and not self.force_bland else ""
            self._record(phase, "pivot", pricing, entering=k, leaving=r, message=msg)
            self.pivot(r, k)
            self._count(phase)
            degenerate_run = degenerate_run + 1 if step <= self.tol_feas else 0
            if not self.force_bland:
                bland = degenerate_run >= self.cycle_guard
                self.bland_used |= bland
            if self.iterations >= self.max_iter:
                return "iteration_limit"

    def run_dual(self, phase: str) -> str:
        """Dual simplex iterations (the basis must be dual feasible)."""
        m, N = self.m, self.N
        degenerate_run = 0
        bland = self.force_bland
        while True:
            rhs = self.T[:m, N]
            art_basic = self.sf.is_art[self.basis] if m else np.zeros(0, dtype=bool)
            # artificial variables must be exactly 0, all others >= 0
            infeas = np.where(art_basic, np.abs(rhs), -rhs) if m else rhs
            cand = np.flatnonzero(infeas > self.tol_feas)
            if cand.size == 0:
                self._record(phase, "optimal", "dual")
                return "optimal"
            if bland:
                r = int(min(cand, key=lambda i: self.basis[i]))
            elif self.pricing == "steepest-edge":
                rows_binv = self.T[cand][:, self.sf.identity_col]   # rows of B^-1
                weights = (np.einsum("ij,ij->i", rows_binv, rows_binv) if not self.exact
                           else (rows_binv * rows_binv).sum(axis=1))
                r = int(cand[np.argmax(infeas[cand] ** 2 / weights)])
            else:
                r = int(cand[np.argmax(infeas[cand])])
            row = self.T[r, :N]
            d = self.T[m, :N]
            elig = self.eligible & ~self.is_basic
            if rhs[r] < 0:
                mask, denom = elig & (row < -self.tol_piv), -row
            else:
                mask, denom = elig & (row > self.tol_piv), row
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                self.infeasible_row = r
                self._record(phase, "infeasible", "dual", leaving=r)
                return "infeasible"
            dd = d[idx] if self.exact else np.maximum(d[idx], 0.0)
            ratios = dd / denom[idx]
            best = ratios.min()
            ties = idx[ratios == best] if self.exact else idx[ratios <= best + self.tol_d]
            if bland:
                k = int(ties.min())
            else:
                k = int(ties[np.argmax(np.abs(row[ties]))])
            msg = "Bland's rule (anti-cycling) is active." if bland and not self.force_bland else ""
            self._record(phase, "pivot", "dual", entering=k, leaving=r, message=msg)
            self.pivot(r, k)
            self._count(phase)
            degenerate_run = degenerate_run + 1 if best <= self.tol_d else 0
            if not self.force_bland:
                bland = degenerate_run >= self.cycle_guard
                self.bland_used |= bland
            if self.iterations >= self.max_iter:
                return "iteration_limit"

    def drive_out_artificials(self, phase: str, pricing: str = "z"):
        """After Phase 1: pivot artificial variables that are basic at zero level out of
        the basis.  If a row has no usable entry the constraint is redundant."""
        m, N = self.m, self.N
        for r in range(m):
            kb = self.basis[r]
            if not self.sf.is_art[kb]:
                continue
            row = self.T[r, :N]
            idx = np.flatnonzero(self.eligible & ~self.is_basic & (np.abs(row) > self.tol_piv))
            if idx.size == 0:
                continue  # redundant equation: the artificial stays basic at zero level
            k = int(idx[np.argmax(np.abs(row[idx]))])
            if not self.exact:
                self.T[r, N] = 0.0
            self._record(phase, "drive-out", pricing, entering=k, leaving=r,
                         message=f"Artificial variable {self.sf.names[kb]} is basic at zero level; "
                                 f"pivot it out of the basis.")
            self.pivot(r, k)
            self._count(phase)

    def shift_costs(self):
        """Replace positive internal costs by 0 so that the slack basis is dual feasible."""
        if self.exact:
            shifted = np.array([min(v, self.zero) for v in self.sf.c], dtype=object)
        else:
            shifted = np.minimum(self.sf.c, 0.0)
        self.cost_z = shifted
        self.set_row(self.m, self.cost_z, self.const_z)

    def restore_costs(self):
        self.cost_z = self.sf.c.copy()
        self.set_row(self.m, self.cost_z, self.const_z)


# --------------------------------------------------------------------------- drivers
def choose_method(lp: LinearProgram) -> str:
    """Pick a sensible method for ``method='auto'``.

    * all constraints ``<=`` with ``b >= 0``: plain simplex (``two-phase`` without Phase 1);
    * no equations and the slack basis is dual feasible (e.g. a minimisation with
      non-negative costs and ``>=`` constraints): ``dual-simplex``;
    * otherwise ``two-phase``.
    """
    needs_art = any(
        t == "=" or (t == "<=" and bi < 0) or (t == ">=" and bi > 0)
        for t, bi in zip(lp.types, lp.b)
    )
    if not needs_art:
        return "two-phase"
    if "=" not in lp.types:
        s = 1 if lp.sense == "max" else -1
        ok = all(
            (sign == ">=0" and s * cj <= 0) or (sign == "<=0" and s * cj >= 0)
            or (sign == "free" and cj == 0)
            for cj, sign in zip(lp.c, lp.var_signs)
        )
        if ok:
            return "dual-simplex"
    return "two-phase"


def solve(lp: LinearProgram, method: str = "auto", exact: bool = False, record_steps: bool = False,
          bland: bool = False, tol: float = 1e-9, max_iter: int | None = None,
          pricing: str = "auto"):
    """Solve ``lp`` with the simplex method and return a :class:`SolveResult`.

    Parameters
    ----------
    method : ``"auto"`` (default), ``"two-phase"``, ``"big-m"`` or ``"dual-simplex"``
    exact : use exact fractions instead of floating point numbers
    record_steps : keep a copy of every tableau (for printing the iterations)
    bland : always use Bland's rule (otherwise only when degenerate pivots repeat)
    tol : zero tolerance for floating point arithmetic
    max_iter : iteration limit (default: generous, grows with the problem size)
    pricing : ``"dantzig"`` (textbook rule: most negative z_j - c_j),
        ``"steepest-edge"`` (far fewer iterations on large problems) or
        ``"auto"`` (steepest edge for large float problems, Dantzig otherwise)
    """
    from .analysis import build_result

    key = METHODS.get(str(method).strip().lower().replace(" ", "-"))
    if key is None:
        raise ValueError(f"unknown method {method!r}; choose auto, two-phase, big-m or dual-simplex")
    requested = key
    if key == "auto":
        key = choose_method(lp)

    start = time.perf_counter()
    sf = build_standard_form(lp, key, exact)
    tab = Tableau(sf, tol=tol, bland=bland, max_iter=max_iter, record=record_steps,
                  pricing=pricing)
    m = sf.m

    if key == "two-phase":
        if sf.has_artificials:
            label = "Two-phase simplex method"
            status = tab.run_primal("w", "Phase 1")
            if status == "optimal":
                if tab.value("w") < -tab.tol_feas:
                    status = "infeasible"
                else:
                    tab.drive_out_artificials("Phase 1")
                    status = tab.run_primal("z", "Phase 2")
        else:
            label = "Simplex method"
            status = tab.run_primal("z", "Simplex")
    elif key == "big-m":
        if sf.has_artificials:
            label = "Big-M method"
            status = tab.run_primal("lex", "Big-M")
        else:
            label = "Simplex method (no artificial variables were needed for Big-M)"
            status = tab.run_primal("z", "Simplex")
    else:
        if tab.is_dual_feasible():
            label = "Dual simplex method"
            status = tab.run_dual("Dual simplex")
        else:
            label = "Dual simplex method (cost shifting) + primal simplex"
            tab.shift_costs()
            status = tab.run_dual("Dual simplex (shifted costs)")
            if status == "optimal":
                tab.restore_costs()
                tab.drive_out_artificials("Primal simplex")
                status = tab.run_primal("z", "Primal simplex")
    if status == "optimal" and not exact and m:
        tab.refresh()
    elapsed = time.perf_counter() - start
    return build_result(lp, sf, tab, status, label, requested, key, elapsed)
