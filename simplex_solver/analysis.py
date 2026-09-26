"""Reading results out of the final simplex tableau.

Everything here is computed from the optimal tableau ``B^-1 [A | b]`` and its
objective row, exactly as done by hand in an operations-research course:

* primal solution            x_B = B^-1 b
* shadow prices (dual prices, "worth" of each constraint / resource)
                              y   = c_B^T B^-1      (objective-row entries under the
                                                     starting basic columns)
* reduced costs              d_j = c_j - (A^T y)_j
* range of optimality        for every objective coefficient c_j
* range of feasibility       for every right-hand side b_i
* duality checks             dual objective, dual feasibility, complementary slackness

Sign conventions (the same as Excel Solver's sensitivity report):

* shadow price  y_i = change of the optimal objective value when b_i increases by one unit
* reduced cost  d_j = change of the objective value when x_j is forced up by one unit
  (0 for basic variables; <= 0 for non-basic variables of a max problem and
  >= 0 for non-basic variables of a min problem)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .utils import INF, fmt_num


def _snap(v):
    """Remove floating point noise: 1.9999999999997 -> 2.0,  -3e-16 -> 0.0."""
    if isinstance(v, Fraction) or v is None:
        return v
    v = float(v)
    if v in (INF, -INF):
        return v
    r = round(v)
    if abs(v - r) <= 1e-9 * max(1.0, abs(v)):
        return float(r) + 0.0
    return v


def _isclose(a, b, scale=1.0, tol=1e-7):
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a == b
    return abs(float(a) - float(b)) <= tol * max(1.0, scale)


@dataclass
class VariableResult:
    """One row of the "variables" part of the sensitivity report."""

    name: str
    value: object
    reduced_cost: object
    cost: object                    # objective coefficient c_j
    basic: bool
    sign: str                       # '>=0', '<=0' or 'free'
    allowable_increase: object = None
    allowable_decrease: object = None

    @property
    def cost_range(self):
        """Range of optimality ``(lower, upper)`` for the objective coefficient."""
        if self.allowable_increase is None:
            return None
        return (self.cost - self.allowable_decrease, self.cost + self.allowable_increase)


@dataclass
class ConstraintResult:
    """One row of the "constraints" part of the sensitivity report."""

    name: str
    lhs: object                     # value of the left-hand side at the optimum
    type: str
    rhs: object
    slack: object                   # slack (<=) or surplus (>=); 0 for equations
    shadow_price: object            # dual price / worth of one extra unit of b_i
    allowable_increase: object = None
    allowable_decrease: object = None

    @property
    def binding(self) -> bool:
        return self.slack == 0

    @property
    def rhs_range(self):
        """Range of feasibility ``(lower, upper)`` for the right-hand side."""
        if self.allowable_increase is None:
            return None
        return (self.rhs - self.allowable_decrease, self.rhs + self.allowable_increase)


@dataclass
class SolveResult:
    """Everything the solver found out about a problem."""

    lp: object
    status: str                     # 'optimal', 'infeasible', 'unbounded', 'iteration_limit'
    method: str                     # description of the method that was used
    method_key: str                 # 'two-phase', 'big-m' or 'dual-simplex'
    requested_method: str
    exact: bool
    iterations: int
    phase_iterations: dict
    elapsed: float
    steps: list
    standard_form: object
    final_tableau: np.ndarray
    final_basis: list
    bland_used: bool = False
    objective: object = None
    x: list = None
    shadow_prices: list = None
    reduced_costs: list = None
    variables: list = None
    constraints: list = None
    basis: list = None
    degenerate: bool = False
    degenerate_vars: list = field(default_factory=list)
    alternative: dict = None
    redundant: list = field(default_factory=list)
    unbounded: dict = None
    infeasible_constraints: list = field(default_factory=list)
    message: str = ""

    # --------------------------------------------------------------- convenience
    @property
    def is_optimal(self) -> bool:
        return self.status == "optimal"

    @property
    def solution(self) -> dict:
        """``{variable name: value}``"""
        return dict(zip(self.lp.var_names, self.x)) if self.x is not None else {}

    @property
    def dual_values(self) -> dict:
        """``{constraint name: shadow price}``"""
        if self.shadow_prices is None:
            return {}
        return dict(zip(self.lp.con_names, self.shadow_prices))

    @property
    def dual_solution(self) -> dict:
        """The optimal solution of the dual problem ``{y1: .., y2: ..}`` (= shadow prices)."""
        if self.shadow_prices is None:
            return {}
        names = self.lp.dual().var_names
        return dict(zip(names, self.shadow_prices))

    def duality(self, solve_dual: bool = False, **solve_options) -> "DualityReport":
        """Duality analysis: dual problem, dual solution, strong duality and
        complementary slackness.  With ``solve_dual=True`` the dual problem is
        also solved from scratch by the simplex method as an independent check."""
        return duality_analysis(self, solve_dual=solve_dual, **solve_options)

    def report(self, **options) -> str:
        """Full text report; see :func:`simplex_solver.report.full_report`."""
        from .report import full_report
        return full_report(self, **options)

    def __repr__(self):
        obj = "" if self.objective is None else f", objective={self.objective}"
        return f"<SolveResult {self.status}{obj}, iterations={self.iterations}>"


# --------------------------------------------------------------------------- extraction
def build_result(lp, sf, tab, status, label, requested, key, elapsed) -> SolveResult:
    res = SolveResult(
        lp=lp, status=status, method=label, method_key=key, requested_method=requested,
        exact=sf.exact, iterations=tab.iterations, phase_iterations=dict(tab.phase_iterations),
        elapsed=elapsed, steps=tab.steps or [], standard_form=sf, final_tableau=tab.T.copy(),
        final_basis=list(tab.basis), bland_used=tab.bland_used,
    )
    res.basis = [sf.names[k] for k in tab.basis]
    if status == "optimal":
        _fill_optimal(res, lp, sf, tab)
    elif status == "unbounded":
        _fill_unbounded(res, lp, sf, tab)
    elif status == "infeasible":
        _fill_infeasible(res, lp, sf, tab)
    else:
        res.message = (f"Stopped after {tab.iterations} iterations (iteration limit). "
                       f"Increase max_iter to continue.")
    return res


def _current_x(sf, tab):
    """User variable values of the current basic solution (plus the internal vector)."""
    m, N = sf.m, sf.N
    zero = Fraction(0) if sf.exact else 0.0
    x_int = np.full(N, zero, dtype=tab.T.dtype)
    if m:
        x_int[tab.basis] = tab.T[:m, N]
    if not sf.exact:
        x_int = np.where(np.abs(x_int) <= tab.tol_feas, 0.0, x_int)
    x = []
    for cols in sf.var_cols:
        v = zero
        for k, mult in cols:
            v = v + mult * x_int[k]
        x.append(v if sf.exact else _snap(v))
    return x, x_int


def _direction(sf, basis, col, k):
    """Change of the user variables when non-basic column k increases by one unit."""
    zero = Fraction(0) if sf.exact else 0.0
    dx = [zero] * sf.n
    if sf.kinds[k] == "x":
        dx[sf.col_var[k]] += sf.col_mult[k]
    for r in np.flatnonzero(col):
        kb = basis[r]
        if sf.kinds[kb] == "x":
            dx[sf.col_var[kb]] -= sf.col_mult[kb] * col[r]
    return dx if sf.exact else [_snap(v) for v in dx]


def _fill_optimal(res, lp, sf, tab):
    m, N, n, s = sf.m, sf.N, sf.n, sf.s
    T = tab.T
    exact = sf.exact
    zero = Fraction(0) if exact else 0.0
    snap = (lambda v: v) if exact else _snap

    x, _x_int = _current_x(sf, tab)
    res.x = x
    obj = sf.c0_user
    for cj, xj in zip(sf.c_user, x):
        obj = obj + cj * xj
    res.objective = snap(obj)

    D = T[m, :N]
    pi = T[m, sf.identity_col] if m else []
    res.shadow_prices = [snap(s * sf.row_sign[i] * pi[i]) for i in range(m)]
    row_of = {k: r for r, k in enumerate(tab.basis)}

    # ----- variables: reduced costs and ranges of optimality
    variables, rcs = [], []
    Dc = D if exact else np.maximum(D, 0.0)
    elig = tab.eligible & ~tab.is_basic
    for j in range(n):
        cols = sf.var_cols[j]
        basic = any(k in row_of for k, _ in cols)
        k0, mult0 = cols[0]
        rc = zero if basic else snap(-s * mult0 * D[k0])
        rcs.append(rc)
        # g = change of the reduced costs per unit change of c_j
        g = np.full(N, zero, dtype=T.dtype)
        for k, mult in cols:
            if k in row_of:
                g = g + (s * mult) * T[row_of[k], :N]
            else:
                g[k] -= s * mult
        lo, hi = -INF, INF
        pos = np.flatnonzero(elig & (g > tab.tol_piv))
        if pos.size:
            lo = (-Dc[pos] / g[pos]).max()
        neg = np.flatnonzero(elig & (g < -tab.tol_piv))
        if neg.size:
            hi = (Dc[neg] / -g[neg]).min()
        variables.append(VariableResult(
            name=lp.var_names[j], value=x[j], reduced_cost=rc, cost=snap(sf.c_user[j]),
            basic=basic, sign=lp.var_signs[j],
            allowable_increase=snap(hi), allowable_decrease=snap(-lo) if lo != -INF else INF,
        ))
    res.reduced_costs = rcs
    res.variables = variables

    # ----- constraints: slack, shadow price and ranges of feasibility
    constraints = []
    beta = T[:m, N] if exact else np.maximum(T[:m, N], 0.0)
    art_rows = sf.is_art[tab.basis] if m else np.zeros(0, dtype=bool)
    lhs_all = sf.A_user @ np.array(x, dtype=T.dtype) if m else []
    for i in range(m):
        lhs = snap(lhs_all[i])
        bi = sf.b_user[i]
        t = lp.types[i]
        slack = snap(bi - lhs) if t == "<=" else snap(lhs - bi) if t == ">=" else zero
        if not exact and abs(slack) <= tab.tol_feas:
            slack = 0.0
        g = sf.row_sign[i] * T[:m, sf.identity_col[i]]
        lo, hi = -INF, INF
        pos = np.flatnonzero(~art_rows & (g > tab.tol_piv))
        if pos.size:
            lo = (-beta[pos] / g[pos]).max()
        neg = np.flatnonzero(~art_rows & (g < -tab.tol_piv))
        if neg.size:
            hi = (beta[neg] / -g[neg]).min()
        if np.any(art_rows & (np.abs(g) > tab.tol_piv)):
            lo, hi = max(lo, zero), min(hi, zero)
        constraints.append(ConstraintResult(
            name=lp.con_names[i], lhs=lhs, type=t, rhs=snap(bi), slack=slack,
            shadow_price=res.shadow_prices[i],
            allowable_increase=snap(hi), allowable_decrease=snap(-lo) if lo != -INF else INF,
        ))
    res.constraints = constraints

    # ----- degeneracy, redundancy and alternative optima
    # An artificial variable that is still basic in a row whose other entries are all 0
    # marks a redundant (linearly dependent) equation; any other basic variable at
    # zero level makes the solution degenerate.
    elig_cols = np.flatnonzero(tab.eligible)
    for r, k in enumerate(tab.basis):
        if sf.is_art[k]:
            row = T[r, elig_cols]
            if not np.any(row != 0 if exact else np.abs(row) > tab.tol_piv):
                res.redundant.append(lp.con_names[sf.col_var[k]])
                continue
        if T[r, N] == 0 if exact else abs(T[r, N]) <= tab.tol_feas:
            res.degenerate_vars.append(sf.names[k])
    res.degenerate = bool(res.degenerate_vars)
    res.alternative = _alternative(sf, tab, x)
    res.message = "Optimal solution found."


def _alternative(sf, tab, x):
    """Look for a different optimal solution (a non-basic variable with zero reduced cost)."""
    m, N = sf.m, sf.N
    T = tab.T
    D = T[m, :N]
    zero_rc = (D == 0) if sf.exact else (np.abs(D) <= tab.tol_d)
    candidates = np.flatnonzero(tab.eligible & ~tab.is_basic & zero_rc)
    found_degenerate = None
    for k in candidates:
        col = T[:m, k]
        dx = _direction(sf, tab.basis, col, k)
        if all((v == 0) if sf.exact else abs(v) <= 1e-9 for v in dx):
            continue
        pos = np.flatnonzero(col > tab.tol_piv)
        if pos.size == 0:
            return {"entering": sf.names[k], "ray": True, "direction": dx, "x": None}
        theta = (T[pos, N] / col[pos]).min()
        if theta <= tab.tol_feas:
            found_degenerate = found_degenerate or sf.names[k]
            continue
        x_alt = [xj + theta * dj for xj, dj in zip(x, dx)]
        if not sf.exact:
            x_alt = [_snap(v) for v in x_alt]
        return {"entering": sf.names[k], "ray": False, "direction": dx, "x": x_alt,
                "theta": theta if sf.exact else _snap(theta)}
    if found_degenerate:
        return {"entering": found_degenerate, "ray": False, "direction": None, "x": None,
                "degenerate": True}
    return None


def _fill_unbounded(res, lp, sf, tab):
    k = tab.unbounded_col
    m = sf.m
    x, _ = _current_x(sf, tab)
    col = tab.T[:m, k]
    dx = _direction(sf, tab.basis, col, k)
    rate = -sf.s * tab.T[m, k]
    res.x = x
    res.unbounded = {"entering": sf.names[k], "x0": x, "direction": dx,
                     "rate": rate if sf.exact else _snap(rate)}
    better = "increase" if lp.sense == "max" else "decrease"
    res.message = (f"The objective is unbounded: {sf.names[k]} can enter the basis but no "
                   f"variable leaves (no positive entry in its column), so z can {better} forever.")


def _fill_infeasible(res, lp, sf, tab):
    m, N = sf.m, sf.N
    T = tab.T
    names = []
    if sf.method in ("two-phase", "big-m") or tab.infeasible_row is None:
        for r, kb in enumerate(tab.basis):
            if sf.is_art[kb] and T[r, N] > tab.tol_feas:
                names.append(lp.con_names[sf.col_var[kb]])
        total = -T[m + 1, N]
        total = total if sf.exact else _snap(total)
        res.message = (f"The problem is infeasible: the artificial variables cannot all be "
                       f"driven to zero (minimum sum of artificials = {fmt_num(total)} > 0).")
    else:
        r = tab.infeasible_row
        kb = tab.basis[r]
        if sf.kinds[kb] != "x":
            names.append(lp.con_names[sf.col_var[kb]])
        res.message = (f"The problem is infeasible: in the dual simplex, row of the basic "
                       f"variable {sf.names[kb]} has a violated value but no possible entering "
                       f"variable (the dual problem is unbounded).")
    res.infeasible_constraints = names


# --------------------------------------------------------------------------- duality
@dataclass
class DualityReport:
    primal: object                  # LinearProgram
    dual: object                    # LinearProgram (the dual problem)
    status: str                     # status of the primal problem
    y: list = None                  # optimal dual solution read from the primal tableau
    primal_objective: object = None
    dual_objective: object = None
    strong_duality: bool = None
    dual_rows: list = None          # per dual constraint: dict(name, lhs, type, rhs, slack, ok)
    sign_ok: list = None            # per dual variable: bool
    cs_constraints: list = None     # per primal constraint: dict(name, y, slack, product, ok)
    cs_variables: list = None       # per primal variable: dict(name, x, dual_slack, product, ok)
    dual_result: object = None      # SolveResult of the dual when solved separately
    recovered_x: list = None        # primal solution read from the dual's shadow prices
    conclusion: str = ""

    @property
    def complementary_slackness(self) -> bool:
        if self.cs_constraints is None:
            return None
        return all(r["ok"] for r in self.cs_constraints + self.cs_variables)

    @property
    def dual_feasible(self) -> bool:
        if self.dual_rows is None:
            return None
        return all(r["ok"] for r in self.dual_rows) and all(self.sign_ok)


def duality_analysis(result: SolveResult, solve_dual: bool = False, **solve_options) -> DualityReport:
    lp = result.lp
    dual = lp.dual()
    rep = DualityReport(primal=lp, dual=dual, status=result.status)
    exact = result.exact
    zero = Fraction(0) if exact else 0.0
    snap = (lambda v: v) if exact else _snap
    sf = result.standard_form
    scale = 1.0
    if not exact:
        vals = [abs(float(v)) for v in list(sf.c_user) + list(sf.b_user)]
        scale = max([1.0] + vals)

    if result.status == "optimal":
        y = list(result.shadow_prices)
        rep.y = y
        rep.primal_objective = result.objective
        w = sf.c0_user
        for bi, yi in zip(sf.b_user, y):
            w = w + bi * yi
        rep.dual_objective = snap(w)
        rep.strong_duality = _isclose(rep.dual_objective, rep.primal_objective, scale)

        rows = []
        dtype = object if exact else float
        dual_lhs = (sf.A_user.T @ np.array(y, dtype=dtype) if lp.num_constraints
                    else np.full(lp.num_vars, zero, dtype=dtype))
        for j in range(lp.num_vars):
            lhs = snap(dual_lhs[j])
            rhs = sf.c_user[j]
            t = dual.types[j]
            slack = snap(lhs - rhs) if t == ">=" else snap(rhs - lhs) if t == "<=" else zero
            ok = (slack >= 0 or _isclose(slack, 0, scale)) and (t != "=" or _isclose(lhs, rhs, scale))
            rows.append({"name": lp.var_names[j], "lhs": lhs, "type": t, "rhs": snap(rhs),
                         "slack": slack if t != "=" else snap(lhs - rhs), "ok": ok})
        rep.dual_rows = rows
        sign_ok = []
        for yi, sgn in zip(y, dual.var_signs):
            if sgn == ">=0":
                sign_ok.append(yi >= 0 or _isclose(yi, 0, scale))
            elif sgn == "<=0":
                sign_ok.append(yi <= 0 or _isclose(yi, 0, scale))
            else:
                sign_ok.append(True)
        rep.sign_ok = sign_ok

        rep.cs_constraints = []
        for con, yi in zip(result.constraints, y):
            prod = snap(yi * con.slack)
            rep.cs_constraints.append({"name": con.name, "y": yi, "slack": con.slack,
                                       "product": prod, "ok": _isclose(prod, 0, scale)})
        rep.cs_variables = []
        for var, row in zip(result.variables, rows):
            ds = row["slack"] if row["type"] != "=" else zero
            prod = snap(var.value * ds)
            rep.cs_variables.append({"name": var.name, "x": var.value, "dual_slack": ds,
                                     "product": prod, "ok": _isclose(prod, 0, scale)})
        rep.conclusion = (
            "Strong duality holds: the optimal values of the primal and the dual are equal."
            if rep.strong_duality else
            "WARNING: primal and dual objective values differ (numerical trouble?).")
    elif result.status == "unbounded":
        rep.conclusion = ("The primal is unbounded, so by weak duality the dual problem has no "
                          "feasible solution (it is infeasible).")
    elif result.status == "infeasible":
        rep.conclusion = ("The primal is infeasible, so the dual is either unbounded or "
                          "infeasible.")
    else:
        rep.conclusion = "The primal was not solved to optimality."

    if solve_dual:
        opts = {"exact": exact}
        opts.update(solve_options)
        dres = dual.solve(**opts)
        rep.dual_result = dres
        if dres.status == "optimal":
            # the shadow prices of the dual are the optimal values of the primal variables
            rep.recovered_x = list(dres.shadow_prices)
            if result.status == "optimal":
                rep.strong_duality = rep.strong_duality and _isclose(dres.objective,
                                                                     result.objective, scale)
        if result.status == "infeasible":
            if dres.status == "unbounded":
                rep.conclusion = "The primal is infeasible and the dual is unbounded."
            elif dres.status == "infeasible":
                rep.conclusion = "Both the primal and the dual problem are infeasible."
    return rep
