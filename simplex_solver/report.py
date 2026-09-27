"""Plain-text reports: problem statement, tableau iterations, solution,
sensitivity analysis and duality analysis."""

from __future__ import annotations

from .utils import fmt_bigm, fmt_linear, fmt_num, fmt_term, heading, text_table

MAX_TABLEAU_COLS = 40      # wider tableaux are summarised instead of printed
MAX_TABLEAU_ROWS = 60
MAX_FORMULATION = 60       # print the formulation only for problems up to this many vars/rows


def _f(x, digits):
    return fmt_num(x, digits)


# --------------------------------------------------------------------------- problem
def format_problem(lp, digits=4, exact=None) -> str:
    """Problem statement; numbers follow the arithmetic (``exact``: fractions,
    otherwise decimals)."""
    if lp.num_vars <= MAX_FORMULATION and lp.num_constraints <= MAX_FORMULATION:
        return lp.formulation(digits, exact=exact)
    return (f"{'Maximize' if lp.sense == 'max' else 'Minimize'} {lp.objective_name} with "
            f"{lp.num_vars} variables and {lp.num_constraints} constraints "
            f"(too large to print; types: {lp.types.count('<=')} '<=', "
            f"{lp.types.count('>=')} '>=', {lp.types.count('=')} '=').")


def format_standard_form(result, digits=4) -> str:
    sf = result.standard_form
    lp = result.lp
    lines = []
    for note in sf.notes:
        lines.append(f"  * {note}")
    if lines:
        lines.insert(0, "Conversions:")
        lines.append("")
    s = sf.s
    obj_coefs = [s * v for v in sf.c]
    names = list(sf.names)
    head = "max" if lp.sense == "max" else "min"
    obj = fmt_linear(obj_coefs[: sum(1 for k in sf.kinds if k == 'x')], names, digits,
                     sf.c0_user)
    slack_names = [nm for nm, kd in zip(names, sf.kinds) if kd in "se"]
    if slack_names:
        obj += " + " + " + ".join(fmt_term(0, nm) for nm in slack_names)
    art_names = [nm for nm, kd in zip(names, sf.kinds) if kd == "a"]
    if art_names and result.method_key == "big-m":
        sign = "-" if lp.sense == "max" else "+"
        obj += " " + " ".join(f"{sign} M {nm}" for nm in art_names)
    present = [label for kind, label in (("s", "slack s"), ("e", "surplus e"), ("a", "artificial a"))
               if kind in sf.kinds]
    lines.append("Standard form" + (f" ({', '.join(present)})" if present else "") + ":")
    lines.append(f"  {head} {lp.objective_name} = {obj}")
    lines.append("  subject to")
    for i in range(sf.m):
        lhs = fmt_linear(sf.A[i], names, digits)
        lines.append(f"    {lhs} = {_f(sf.b[i], digits)}")
    lines.append(f"    {', '.join(names)} >= 0")
    if art_names and result.method_key == "two-phase":
        lines.append("")
        lines.append(f"Phase 1 objective:  min r = {' + '.join(art_names)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- tableaux
def _display_rows(step, sf, lp):
    """Return (label, entries, value, rule) of the objective row as shown to the user."""
    m, N = sf.m, sf.N
    T = step.T
    s = sf.s
    if step.pricing == "w":
        return ("r", [-T[m + 1, k] for k in range(N)], -T[m + 1, N], "min")
    if step.pricing == "lex":
        return ("z", [(s * T[m + 1, k], s * T[m, k]) for k in range(N)],
                (s * T[m + 1, N], s * T[m, N]), lp.sense)
    return (lp.objective_name, [s * T[m, k] for k in range(N)], s * T[m, N], lp.sense)


def format_tableau(step, result, digits=4) -> str:
    sf = result.standard_form
    lp = result.lp
    m, N = sf.m, sf.N
    T = step.T
    label, entries, value, rule = _display_rows(step, sf, lp)
    lex = step.pricing == "lex"

    def cell(v):
        return fmt_bigm(v[0], v[1], digits) if lex else _f(v, digits)

    primal_ratio = step.kind in ("pivot", "unbounded") and step.pricing != "dual" \
        and step.entering is not None
    headers = ["Basis"] + list(sf.names) + ["RHS"] + (["Ratio"] if primal_ratio else [])
    rows = []
    for r in range(m):
        row = [sf.names[step.basis[r]]] + [_f(T[r, k], digits) for k in range(N)] + [_f(T[r, N], digits)]
        if primal_ratio:
            a = T[r, step.entering]
            if a > 0 and (not isinstance(a, float) or a > 1e-12):
                ratio = _f(max(T[r, N], 0) / a if not sf.exact else T[r, N] / a, digits)
                row.append(ratio + ("  <- min" if r == step.leaving else ""))
            else:
                row.append("-")
        elif step.pricing == "dual" and r == step.leaving:
            row[-1] += "  <- leaves"
        rows.append(row)
    obj_row = [label] + [cell(v) for v in entries] + [cell(value)]
    if primal_ratio:
        obj_row.append("")
    rows.append(obj_row)
    if step.pricing == "dual" and step.kind == "pivot":
        r = step.leaving
        ratio_row = ["ratio"]
        neg = T[r, N] < 0
        for k in range(N):
            a = T[r, k]
            ok = (a < 0 if neg else a > 0) and not sf.is_art[k] and k not in step.basis
            if ok and (sf.exact or abs(a) > 1e-12):
                d = abs(entries[k])
                ratio_row.append(_f(d / abs(a), digits) + ("*" if k == step.entering else ""))
            else:
                ratio_row.append("")
        ratio_row.append("")
        rows.append(ratio_row)
    table = text_table(headers, rows, align="l" + "r" * (len(headers) - 1))
    # separator line above the objective row
    lines = table.split("\n")
    sep = lines[1]
    insert_at = 2 + m
    lines.insert(insert_at, sep)
    return "\n".join(lines)


def _rule_text(rule, lex=False):
    if rule == "max":
        return "all z_j - c_j >= 0" + (" (compare the M parts first)" if lex else "")
    return "all z_j - c_j <= 0" + (" (compare the M parts first)" if lex else "")


def describe_step(step, result, digits=4) -> str:
    sf = result.standard_form
    N = sf.N
    T = step.T
    label, entries, value, rule = _display_rows(step, sf, result.lp)
    lex = step.pricing == "lex"
    names = sf.names
    out = []
    if step.kind == "pivot" and step.pricing != "dual":
        k, r = step.entering, step.leaving
        e = entries[k]
        e_txt = fmt_bigm(e[0], e[1], digits) if lex else _f(e, digits)
        word = "most negative" if rule == "max" else "most positive"
        if "Bland" in step.message:
            word = "Bland's rule: first improving"
        out.append(f"Entering variable: {names[k]}  ({word} z_j - c_j = {e_txt})")
        out.append(f"Leaving variable:  {names[step.basis[r]]}  "
                   f"(minimum ratio {_f(T[r, N] / T[r, k], digits)})")
        out.append(f"Pivot element: {_f(T[r, k], digits)}")
    elif step.kind == "pivot":
        k, r = step.entering, step.leaving
        if sf.is_art[step.basis[r]]:
            why = f"artificial variable = {_f(T[r, N], digits)}, but it must be 0"
        else:
            why = f"most negative right-hand side = {_f(T[r, N], digits)}"
        out.append(f"Leaving variable:  {names[step.basis[r]]}  ({why})")
        out.append(f"Entering variable: {names[k]}  (minimum ratio |z_j - c_j| / |a_rj| = "
                   f"{_f(abs(entries[k]) / abs(T[r, k]), digits)})")
        out.append(f"Pivot element: {_f(T[r, k], digits)}")
    elif step.kind == "drive-out":
        out.append(step.message)
        out.append(f"Entering variable: {names[step.entering]}; leaving variable: "
                   f"{names[step.basis[step.leaving]]}; pivot element: "
                   f"{_f(T[step.leaving, step.entering], digits)}")
        return "\n".join(out)
    elif step.kind == "optimal":
        if step.pricing == "w":
            out.append(f"Phase 1 is optimal ({_rule_text('min')}) with r = {_f(value, digits)}.")
            if fmt_num(value, digits) == "0":
                out.append("r = 0, so a basic feasible solution of the original problem was found; "
                           "the artificial variables are removed for Phase 2.")
            else:
                out.append("r > 0: the artificial variables cannot all be zero, so the original "
                           "problem has no feasible solution (infeasible).")
        elif step.pricing == "dual":
            if "shifted" in step.phase:
                out.append("All basic variables are feasible: a feasible basis was found. The "
                           "original costs are restored and the primal simplex continues.")
            else:
                out.append("All basic variables are feasible (RHS >= 0) and the tableau is dual "
                           "feasible: optimal.")
        else:
            out.append(f"Optimal: {_rule_text(rule, lex)}.")
    elif step.kind == "unbounded":
        out.append(f"Entering variable {names[step.entering]} has no positive entry in its "
                   f"column: the objective is unbounded.")
    elif step.kind == "infeasible":
        if step.message:
            out.append(step.message)
        elif step.pricing == "dual":
            out.append(f"Row of {names[step.basis[step.leaving]]} is infeasible and has no "
                       f"suitable entering variable: the problem is infeasible.")
    if step.message and step.kind == "pivot":
        out.append(step.message)
    return "\n".join(out)


def format_steps(result, digits=4) -> str:
    sf = result.standard_form
    if not result.steps:
        return "(no iterations were recorded; solve with record_steps=True)"
    big = sf.N + 2 > MAX_TABLEAU_COLS or sf.m > MAX_TABLEAU_ROWS
    lines = []
    if big:
        lines.append(f"(tableau is {sf.m} x {sf.N}: too large to print, showing a summary of "
                     f"each iteration)")
    pivots = 0
    phase = None
    for step in result.steps:
        new_phase = step.phase != phase
        if new_phase:
            phase = step.phase
            lines.append("")
            lines.append(f"--- {phase} ---")
        label, entries, value, rule = _display_rows(step, sf, result.lp)
        val_txt = fmt_bigm(value[0], value[1], digits) if step.pricing == "lex" else _f(value, digits)
        title = f"Tableau {pivots}"
        if pivots == 0:
            title += " (initial)"
        elif new_phase:
            title += f" (start of {phase})"
        if step.kind == "optimal":
            title += " - end of Phase 1" if step.pricing == "w" else " - optimal"
        elif step.kind in ("unbounded", "infeasible"):
            title += f" - {step.kind}"
        if step.kind in ("pivot", "drive-out"):
            pivots += 1
        if big:
            desc = describe_step(step, result, digits).replace("\n", "; ")
            lines.append(f"{title}: {label} = {val_txt}. {desc}")
            continue
        lines.append("")
        lines.append(f"{title}   [{label} = {val_txt}]")
        lines.append(format_tableau(step, result, digits))
        desc = describe_step(step, result, digits)
        if desc:
            lines.append(desc)
    return "\n".join(lines)


# --------------------------------------------------------------------------- results
def format_solution(result, digits=4) -> str:
    lp = result.lp
    lines = [f"Status:     {result.status.upper()}",
             f"Method:     {result.method}",
             f"Iterations: {result.iterations}" + (
                 "  (" + ", ".join(f"{k}: {v}" for k, v in result.phase_iterations.items()) + ")"
                 if len(result.phase_iterations) > 1 else ""),
             f"Arithmetic: {'exact fractions' if result.exact else 'floating point'}",
             f"Time:       {result.elapsed:.4f} s"]
    if result.status == "optimal":
        lines.append("")
        lines.append(f"Optimal objective value:  {lp.objective_name}* = {_f(result.objective, digits)}")
        lines.append("")
        nonzero_only = lp.num_vars > 100
        rows = [[v.name, _f(v.value, digits), "basic" if v.basic else "non-basic"]
                for v in result.variables if not nonzero_only or fmt_num(v.value, digits) != "0"]
        title = "Optimal solution" + (" (non-zero variables only)" if nonzero_only else "")
        lines.append(title + ":")
        lines.append(text_table(["Variable", "Value", "Status"], rows, align="lrl"))
        lines.append("")
        lines.append("Basic variables in the final tableau: " + ", ".join(result.basis))
    else:
        lines.append("")
        lines.append(result.message)
        if result.status == "unbounded" and result.unbounded:
            u = result.unbounded
            x0 = ", ".join(f"{n} = {_f(v, digits)}" for n, v in zip(lp.var_names, u["x0"]))
            d = ", ".join(f"{n}: {_f(v, digits)}" for n, v in zip(lp.var_names, u["direction"]))
            lines.append(f"Along the ray  x(t) = x0 + t*d  (t >= 0) the objective changes by "
                         f"{_f(u['rate'], digits)} per unit of t, where")
            lines.append(f"  x0 = ({x0})")
            lines.append(f"  d  = ({d})")
        if result.status == "infeasible" and result.infeasible_constraints:
            lines.append("Constraints involved in the conflict: "
                         + ", ".join(result.infeasible_constraints))
    return "\n".join(lines)


def format_sensitivity(result, digits=4) -> str:
    if result.status != "optimal":
        return "(sensitivity analysis is only available for an optimal solution)"
    f = lambda v: _f(v, digits)  # noqa: E731
    lines = ["Decision variables (reduced cost = change in the objective per unit increase of",
             "the variable; range of optimality = objective coefficients for which the current",
             "basis stays optimal):", ""]
    rows = []
    for v in result.variables:
        lo, hi = v.cost_range
        rows.append([v.name, f(v.value), f(v.reduced_cost), f(v.cost),
                     f(v.allowable_increase), f(v.allowable_decrease), f"[{f(lo)}, {f(hi)}]"])
    lines.append(text_table(["Variable", "Value", "Reduced cost", "Obj. coef",
                             "Allow. increase", "Allow. decrease", "Range of optimality"], rows))
    lines += ["", "Constraints (shadow price = worth of one more unit of the right-hand side;",
              "range of feasibility = right-hand sides for which the shadow price stays valid):", ""]
    rows = []
    for c in result.constraints:
        lo, hi = c.rhs_range
        kind = "slack" if c.type == "<=" else "surplus" if c.type == ">=" else ""
        slack = f"{f(c.slack)} {kind}".strip() if c.type != "=" else "0"
        rows.append([c.name, f(c.lhs), c.type, f(c.rhs), slack,
                     "binding" if c.binding else "not binding", f(c.shadow_price),
                     f(c.allowable_increase), f(c.allowable_decrease), f"[{f(lo)}, {f(hi)}]"])
    lines.append(text_table(["Constraint", "LHS", "Type", "RHS", "Slack/Surplus", "Status",
                             "Shadow price", "Allow. increase", "Allow. decrease",
                             "Range of feasibility"], rows, align="lrlrrlrrrl"))
    return "\n".join(lines)


def format_notes(result, digits=4) -> str:
    lp = result.lp
    notes = []
    if result.status == "optimal":
        alt = result.alternative
        if alt:
            if alt.get("ray"):
                notes.append(f"Alternative optima: non-basic {alt['entering']} has a zero reduced "
                             f"cost and can increase without limit, so there are infinitely many "
                             f"optimal solutions along a ray.")
            elif alt.get("x") is not None:
                xs = ", ".join(f"{n} = {_f(v, digits)}" for n, v in zip(lp.var_names, alt["x"]))
                notes.append(f"Alternative optima: non-basic {alt['entering']} has a zero reduced "
                             f"cost. Bringing it into the basis gives another optimal solution "
                             f"({xs}); every point on the segment between the two is optimal too.")
            else:
                notes.append(f"Non-basic {alt['entering']} has a zero reduced cost (a degenerate "
                             f"pivot): other optimal bases exist and the optimum may not be unique.")
        else:
            notes.append("The optimal solution is unique (every non-basic variable has a non-zero "
                         "reduced cost)." if not result.degenerate else
                         "No non-basic variable has a zero reduced cost.")
        if result.degenerate:
            notes.append("Degenerate solution: basic variable(s) " + ", ".join(result.degenerate_vars)
                         + " equal 0. Shadow prices may not be unique and some ranges can be 0 on "
                           "one side.")
        if result.redundant:
            notes.append("Redundant constraint(s) (linear combinations of the others): "
                         + ", ".join(result.redundant) + ".")
    if result.bland_used:
        notes.append("Degenerate pivots repeated, so Bland's anti-cycling rule was used for some "
                     "iterations.")
    if not notes:
        return ""
    return "\n".join(f"* {n}" for n in notes)


def format_duality(result, dual_report=None, digits=4) -> str:
    rep = dual_report or result.duality()
    lp = result.lp
    dual = rep.dual
    f = lambda v: _f(v, digits)  # noqa: E731
    lines = ["Dual problem:"]
    lines.append("\n".join("  " + ln for ln in format_problem(dual, digits, result.exact).split("\n")))
    lines.append("")
    lines.append("(dual variable y_i belongs to primal constraint i; dual constraint j belongs "
                 "to primal variable j)")
    if any(not nm.startswith("C") for nm in lp.con_names):
        lines.append("  " + ", ".join(f"{yn} <-> {cn}" for yn, cn in zip(dual.var_names, lp.con_names)))
    lines.append("")
    if rep.y is not None:
        lines.append("Optimal dual solution, read from the final primal tableau "
                     "(y = c_B B^-1 = shadow prices):")
        lines.append("  " + (", ".join(f"{yn} = {f(v)}" for yn, v in zip(dual.var_names, rep.y))
                             or "(the primal has no constraints, so the dual has no variables)"))
        lines.append(f"  Dual objective  {dual.objective_name}* = {f(rep.dual_objective)}   "
                     f"Primal objective  {lp.objective_name}* = {f(rep.primal_objective)}")
        lines.append("  " + rep.conclusion)
        lines.append("")
        lines.append("Dual feasibility check (dual constraint j is satisfied; its slack equals "
                     "|reduced cost of x_j|):")
        rows = [[r["name"], f(r["lhs"]), r["type"], f(r["rhs"]), f(r["slack"]),
                 "ok" if r["ok"] else "VIOLATED"] for r in rep.dual_rows]
        lines.append(text_table(["Dual constraint", "LHS", "Type", "RHS", "Slack", "Check"], rows,
                                align="lrlrrl"))
        bad_signs = [yn for yn, ok in zip(dual.var_names, rep.sign_ok) if not ok]
        lines.append("  Sign restrictions of the dual variables: "
                     + ("ok" if not bad_signs else "VIOLATED for " + ", ".join(bad_signs)))
        lines.append("")
        lines.append("Complementary slackness:")
        rows = [[r["name"], f(r["y"]), f(r["slack"]), f(r["product"]), "ok" if r["ok"] else "FAIL"]
                for r in rep.cs_constraints]
        if rows:
            lines.append(text_table(["Primal constraint", "y_i", "slack_i", "y_i * slack_i", "Check"],
                                    rows, align="lrrrl"))
            lines.append("")
        rows = [[r["name"], f(r["x"]), f(r["dual_slack"]), f(r["product"]), "ok" if r["ok"] else "FAIL"]
                for r in rep.cs_variables]
        lines.append(text_table(["Primal variable", "x_j", "dual slack_j", "x_j * slack_j", "Check"],
                                rows, align="lrrrl"))
        verdict = "satisfied" if rep.complementary_slackness else "NOT satisfied"
        lines.append(f"  Complementary slackness is {verdict}.")
    else:
        lines.append(rep.conclusion)
    if rep.dual_result is not None:
        d = rep.dual_result
        lines.append("")
        lines.append(f"Dual problem solved independently ({d.method}, {d.iterations} iterations): "
                     f"status {d.status.upper()}")
        if d.status == "optimal":
            lines.append("  " + ", ".join(f"{n} = {f(v)}" for n, v in zip(dual.var_names, d.x)))
            lines.append(f"  {dual.objective_name}* = {f(d.objective)}")
            if rep.recovered_x is not None:
                lines.append("  Primal solution read from the dual's shadow prices: "
                             + ", ".join(f"{n} = {f(v)}" for n, v in zip(lp.var_names, rep.recovered_x)))
            if result.status == "optimal":
                same = rep.strong_duality
                lines.append(f"  {'z* = w*: strong duality confirmed.' if same else 'WARNING: z* != w*'}")
        elif rep.conclusion:
            lines.append("  " + rep.conclusion)
    return "\n".join(lines)


def full_report(result, steps: bool = None, sensitivity: bool = True, duality: bool = True,
                solve_dual: bool = False, digits: int = 4, dual_report=None,
                show_problem: bool = True) -> str:
    """Complete report.  ``steps=None`` prints the iterations when they were recorded."""
    parts = []
    if show_problem:
        parts += [heading("PROBLEM"), format_problem(result.lp, digits, result.exact), ""]
    if steps is None:
        steps = bool(result.steps)
    if steps:
        parts += [heading("STANDARD FORM"), format_standard_form(result, digits), ""]
        parts += [heading("SIMPLEX ITERATIONS"), format_steps(result, digits).lstrip("\n"), ""]
    parts += [heading("RESULT"), format_solution(result, digits), ""]
    if result.status == "optimal" and sensitivity:
        parts += [heading("SENSITIVITY ANALYSIS"), format_sensitivity(result, digits), ""]
    if duality:
        if dual_report is None:
            dual_report = result.duality(solve_dual=solve_dual)
        parts += [heading("DUALITY"), format_duality(result, dual_report, digits), ""]
    notes = format_notes(result, digits)
    if notes:
        parts += [heading("NOTES"), notes, ""]
    return "\n".join(parts)
