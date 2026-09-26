"""Read linear programs from text (algebraic form) or JSON (matrix form).

Algebraic text format
---------------------
::

    # comments start with '#' or '//'
    max: 3x1 + 5x2            (also: maximize z = 3x1 + 5x2,  min: ...,  minimize ...)
    subject to                (optional; also 's.t.', 'st', 'such that')
    labor:    x1        <= 4  (an optional name followed by ':')
    material:       2x2 <= 12
              3x1 + 2x2 <= 18
    2 <= x1 + x2 <= 10        (a range becomes two constraints)
    x1, x2 >= 0               (sign declarations; '>= 0' is the default anyway)
    x3 <= 0                   (x3 is a non-positive variable)
    x4 free                   (x4 is unrestricted in sign; also 'urs', 'unrestricted')

Coefficients may be written ``3x1``, ``3 x1``, ``3*x1``, ``1/2 x1``, ``(1/2)x1``,
``x1/2`` or ``2.5e-3 x1``; variables may appear on both sides of a relation.
Relations: ``<=``, ``>=``, ``=`` (also ``<``, ``>``, ``==``, ``=<``, ``=>``, ``≤``, ``≥``).
A line of the form ``x <= 0`` / ``x >= 0`` (a single variable or a list of
variables compared with zero) is read as a *sign restriction*, not a constraint.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

from .model import CONSTRAINT_TYPES, LinearProgram


class ParseError(ValueError):
    """Raised when a problem description cannot be understood."""


_OBJ_RE = re.compile(r"^\s*(maximi[sz]e|minimi[sz]e|maximum|minimum|max|min)\b\s*:?\s*(.*)$", re.I)
_ST_RE = re.compile(r"^\s*(subject\s+to|such\s+that|s\.\s*t\.?|st\b|constraints?\b)\s*:?\s*(.*)$", re.I)
_NAME_PREFIX_RE = re.compile(r"^\s*([A-Za-z_][\w.\-\[\] ]*?)\s*:\s*(.+)$")
_OBJ_NAME_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.*)$")
_END_RE = re.compile(r"^\s*end\s*$", re.I)

_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+(?![A-Za-z_]))?"
_TOKEN_RE = re.compile(
    rf"\s*(?:(?P<num>{_NUM})|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<rel><=|>=|=<|=>|==|<|>|=|≤|≥)|(?P<op>[+\-*/(),]))"
)
_FREE_WORDS = {"free", "urs", "unrestricted"}
_FILLER_WORDS = {"is", "are", "in", "sign", "and", "variable", "variables"}


# ---------------------------------------------------------------- expressions
class _Lin:
    """A linear expression: sum(coefs[name] * name) + const."""

    __slots__ = ("coefs", "const")

    def __init__(self, coefs=None, const=Fraction(0)):
        self.coefs = coefs or {}
        self.const = const

    def is_const(self):
        return all(v == 0 for v in self.coefs.values())

    def __add__(self, other):
        coefs = dict(self.coefs)
        for k, v in other.coefs.items():
            coefs[k] = coefs.get(k, 0) + v
        return _Lin(coefs, self.const + other.const)

    def __neg__(self):
        return _Lin({k: -v for k, v in self.coefs.items()}, -self.const)

    def __sub__(self, other):
        return self + (-other)

    def scale(self, f):
        return _Lin({k: v * f for k, v in self.coefs.items()}, self.const * f)


def _tokenize(text: str):
    text = text.replace("−", "-").replace("×", "*")
    tokens, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        mt = _TOKEN_RE.match(text, pos)
        if not mt or mt.end() == pos:
            raise ParseError(f"unexpected character {text[pos:].strip()[0]!r}")
        kind = mt.lastgroup
        value = mt.group(kind)
        if kind == "rel":
            value = CONSTRAINT_TYPES[value]
        tokens.append((kind, value))
        pos = mt.end()
    return tokens


class _ExprParser:
    """Recursive-descent parser for linear expressions (supports + - * / and parentheses)."""

    def __init__(self, tokens, order):
        self.tokens = tokens
        self.i = 0
        self.order = order  # records the order in which variables are first seen

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        if not self.tokens:
            raise ParseError("empty expression")
        e = self.expr()
        if self.i != len(self.tokens):
            raise ParseError(f"unexpected {self.peek()[1]!r}")
        return e

    def expr(self):
        kind, val = self.peek()
        neg = False
        if kind == "op" and val in "+-":
            self.take()
            neg = val == "-"
        e = self.term()
        if neg:
            e = -e
        while True:
            kind, val = self.peek()
            if kind == "op" and val in "+-":
                self.take()
                t = self.term()
                e = e + t if val == "+" else e - t
            else:
                return e

    def term(self):
        e = self.factor()
        while True:
            kind, val = self.peek()
            if kind == "op" and val in "*/":
                self.take()
                f = self.factor()
                e = self._mul(e, f) if val == "*" else self._div(e, f)
            elif kind in ("num", "name") or (kind == "op" and val == "("):
                e = self._mul(e, self.factor())  # implicit multiplication: 3x1, 2(x1 + x2)
            else:
                return e

    def factor(self):
        kind, val = self.take()
        if kind == "num":
            return _Lin({}, Fraction(val))
        if kind == "name":
            if val not in self.order:
                self.order.append(val)
            return _Lin({val: Fraction(1)})
        if kind == "op" and val in "+-":
            f = self.factor()
            return -f if val == "-" else f
        if kind == "op" and val == "(":
            e = self.expr()
            k2, v2 = self.take()
            if (k2, v2) != ("op", ")"):
                raise ParseError("missing ')'")
            return e
        raise ParseError(f"unexpected {val!r}" if val is not None else "expression ends unexpectedly")

    @staticmethod
    def _mul(a, b):
        if a.is_const():
            return b.scale(a.const)
        if b.is_const():
            return a.scale(b.const)
        raise ParseError("products of variables are not linear")

    @staticmethod
    def _div(a, b):
        if not b.is_const():
            raise ParseError("cannot divide by a variable")
        if b.const == 0:
            raise ParseError("division by zero")
        return a.scale(1 / b.const)


def _split_on(tokens, kind, value=None):
    parts, cur = [], []
    for tok in tokens:
        if tok[0] == kind and (value is None or tok[1] == value):
            parts.append(cur)
            cur = []
        else:
            cur.append(tok)
    parts.append(cur)
    return parts


# ---------------------------------------------------------------- statements
def _single_sign_decl(tokens):
    """``x1, x2 >= 0`` / ``x3 <= 0`` / ``0 <= x1``  ->  (names, sign); otherwise None."""
    rels = [v for k, v in tokens if k == "rel"]
    if len(rels) != 1 or rels[0] == "=":
        return None
    lhs, rhs = _split_on(tokens, "rel")

    def name_list(part):
        if not part or len(part) % 2 == 0:
            return None
        for idx, tok in enumerate(part):
            if idx % 2 == 0 and tok[0] != "name":
                return None
            if idx % 2 == 1 and tok != ("op", ","):
                return None
        return [tok[1] for tok in part[::2]]

    def is_zero(part):
        return len(part) == 1 and part[0][0] == "num" and Fraction(part[0][1]) == 0

    rel = rels[0]
    names = name_list(lhs)
    if names is not None and is_zero(rhs):
        return names, (">=0" if rel == ">=" else "<=0")
    names = name_list(rhs)
    if names is not None and is_zero(lhs):
        return names, ("<=0" if rel == ">=" else ">=0")
    return None


def _sign_declaration(tokens):
    """Return a list of ``(names, sign)`` if the tokens declare variable signs, else None."""
    words = {v.lower() for k, v in tokens if k == "name"}
    if words & _FREE_WORDS:
        names = [v for k, v in tokens if k == "name" and v.lower() not in _FREE_WORDS | _FILLER_WORDS]
        others = [t for t in tokens if t[0] != "name" and t != ("op", ",")]
        if names and not others:
            return [(names, "free")]
        raise ParseError("could not read the 'free' declaration (write e.g. 'x3 free')")
    single = _single_sign_decl(tokens)
    if single is not None:
        return [single]
    if ("op", ",") in tokens:
        decls = [_single_sign_decl(piece) for piece in _split_on(tokens, "op", ",")]
        if all(d is not None for d in decls):
            return decls
        raise ParseError("commas are only allowed in sign declarations like 'x1, x2 >= 0'")
    return None


def _constraints_from(tokens, order):
    """Turn ``e0 rel e1 [rel e2]`` into a list of (coefs, rel, rhs)."""
    rels = [v for k, v in tokens if k == "rel"]
    parts = _split_on(tokens, "rel")
    if len(rels) == 0:
        raise ParseError("expected a relation such as <=, >= or =")
    if len(rels) > 2:
        raise ParseError("too many relations in one constraint")
    exprs = [_ExprParser(p, order).parse() for p in parts]
    out = []
    for (lhs, rhs), rel in zip(zip(exprs, exprs[1:]), rels):
        if lhs.is_const() and not rhs.is_const():
            lhs, rhs = rhs, lhs
            rel = {"<=": ">=", ">=": "<=", "=": "="}[rel]
        diff = lhs - rhs
        coefs = {k: v for k, v in diff.coefs.items() if v != 0}
        if not coefs:
            raise ParseError("constraint contains no variables")
        out.append((coefs, rel, -diff.const))
    return out


def parse_lp(text: str) -> LinearProgram:
    """Parse a problem written in algebraic form (see the module docstring)."""
    order: list[str] = []
    sense = None
    objective = None
    objective_name = "z"
    constraints = []  # (name or None, coefs, rel, rhs, line number)
    signs: dict[str, str] = {}
    pending_objective = False

    statements = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].split("//", 1)[0]
        for piece in line.split(";"):
            if piece.strip():
                statements.append((lineno, piece.strip()))

    for lineno, stmt in statements:
        try:
            if _END_RE.match(stmt):
                break
            if pending_objective:
                pending_objective = False
                mo = _OBJ_NAME_RE.match(stmt)
                if mo and not any(r in mo.group(2) for r in "<>="):
                    objective_name, stmt = mo.group(1), mo.group(2)
                objective = _ExprParser(_tokenize(stmt), order).parse()
                continue
            mo = _OBJ_RE.match(stmt)
            if mo and sense is None:
                sense = "max" if mo.group(1).lower().startswith("max") else "min"
                rest = mo.group(2).strip()
                if not rest:
                    pending_objective = True
                    continue
                mn = _OBJ_NAME_RE.match(rest)
                if mn and not any(r in mn.group(2) for r in "<>="):
                    objective_name, rest = mn.group(1), mn.group(2)
                objective = _ExprParser(_tokenize(rest), order).parse()
                continue
            if mo:
                raise ParseError("only one objective function is allowed")
            ms = _ST_RE.match(stmt)
            if ms:
                stmt = ms.group(2).strip()
                if not stmt:
                    continue
            name = None
            mname = _NAME_PREFIX_RE.match(stmt)
            if mname:
                name, stmt = mname.group(1).strip(), mname.group(2)
            tokens = _tokenize(stmt)
            decls = _sign_declaration(tokens)
            if decls is not None:
                if name is not None:
                    raise ParseError("sign declarations cannot have a name")
                for vars_, sign in decls:
                    for v in vars_:
                        if v not in order:
                            order.append(v)
                        signs[v] = sign
                continue
            parts = _constraints_from(tokens, order)
            for idx, (coefs, rel, rhs) in enumerate(parts):
                cname = name
                if name is not None and len(parts) == 2:
                    cname = f"{name}_{'lo' if idx == 0 else 'hi'}"
                constraints.append((cname, coefs, rel, rhs, lineno))
        except ParseError as exc:
            raise ParseError(f"line {lineno}: {exc}  ->  {stmt!r}") from None

    if sense is None or objective is None:
        raise ParseError("no objective found: start with a line like 'max: 3x1 + 5x2' or 'min: ...'")
    if not order:
        raise ParseError("the problem has no variables")

    col = {v: j for j, v in enumerate(order)}
    n = len(order)
    c = [Fraction(0)] * n
    for v, a in objective.coefs.items():
        c[col[v]] += a
    A, b, types, names = [], [], [], []
    used = set()
    for cname, _coefs, _rel, _rhs, lineno in constraints:
        if cname is not None:
            if cname in used:
                raise ParseError(f"line {lineno}: duplicate constraint name {cname!r}")
            used.add(cname)
    counter = 0
    for cname, coefs, rel, rhs, _lineno in constraints:
        row = [Fraction(0)] * n
        for v, a in coefs.items():
            row[col[v]] += a
        if cname is None:
            counter += 1
            while f"C{counter}" in used:
                counter += 1
            cname = f"C{counter}"
            used.add(cname)
        A.append(row)
        b.append(rhs)
        types.append(rel)
        names.append(cname)
    return LinearProgram(
        c=c, A=A if A else [], b=b, types=types, sense=sense, var_names=order,
        con_names=names, var_signs=[signs.get(v, ">=0") for v in order],
        objective_constant=objective.const, objective_name=objective_name,
    )


# ---------------------------------------------------------------- JSON / dict
def lp_from_dict(data: dict) -> LinearProgram:
    """Build a problem from a dictionary (e.g. loaded from JSON).

    Matrix style::

        {"sense": "max", "c": [3, 5], "A": [[1, 0], [0, 2], [3, 2]], "b": [4, 12, 18],
         "types": ["<=", "<=", "<="], "var_names": ["x1", "x2"], "var_signs": [">=0", ">=0"]}

    Row style::

        {"sense": "min", "objective": [2, 3],
         "constraints": [{"name": "protein", "coefficients": [1, 2], "type": ">=", "rhs": 8}, ...]}

    Or simply ``{"problem": "max: 3x1 + 5x2\\n ..."}`` with the algebraic text.
    """
    if "problem" in data and isinstance(data["problem"], str):
        return parse_lp(data["problem"])
    sense = data.get("sense", data.get("objective_sense", "max"))
    c = data.get("c", data.get("objective"))
    if c is None:
        raise ParseError("JSON problem needs the objective coefficients under 'c' or 'objective'")
    rows = data.get("constraints")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        A = [r.get("coefficients", r.get("coef", r.get("a"))) for r in rows]
        b = [r.get("rhs", r.get("b")) for r in rows]
        types = [r.get("type", r.get("relation", "<=")) for r in rows]
        con_names = [r.get("name") for r in rows]
        con_names = None if any(nm is None for nm in con_names) else con_names
    else:
        A = data.get("A", rows if rows is not None else [])
        b = data.get("b", data.get("rhs", []))
        types = data.get("types", data.get("relations"))
        con_names = data.get("con_names", data.get("constraint_names"))
    return LinearProgram(
        c=c, A=A, b=b, types=types, sense=sense,
        var_names=data.get("var_names", data.get("variables")),
        con_names=con_names,
        var_signs=data.get("var_signs", data.get("signs")),
        objective_constant=data.get("objective_constant", 0),
    )


def read_problem(path) -> LinearProgram:
    """Read a problem file: ``.json`` (matrix form) or anything else (algebraic text)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    if p.suffix.lower() == ".json" or text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"{p.name}: invalid JSON ({exc})") from None
        lp = lp_from_dict(data)
    else:
        lp = parse_lp(text)
    if lp.name is None:
        lp.name = p.stem
    return lp
