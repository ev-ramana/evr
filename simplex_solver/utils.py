"""Number conversion and text-formatting helpers shared by the whole package."""

from __future__ import annotations

import math
import numbers
from fractions import Fraction

INF = float("inf")


def to_fraction(value) -> Fraction:
    """Convert an int / float / str / Fraction to an exact :class:`Fraction`.

    Floats are converted through their shortest decimal representation, so
    ``0.1`` becomes ``1/10`` (and not the binary value 3602879701896397/2**55).
    Strings may be integers, decimals or fractions such as ``"3/4"``.
    """
    if isinstance(value, Fraction):
        return value
    if isinstance(value, numbers.Integral):
        return Fraction(int(value))
    if isinstance(value, str):
        return Fraction(value.strip())
    if isinstance(value, numbers.Rational):
        return Fraction(int(value.numerator), int(value.denominator))
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"coefficients must be finite numbers, got {value!r}")
    return Fraction(repr(f))


def is_inf(x) -> bool:
    return isinstance(x, float) and math.isinf(x)


def fmt_num(x, digits: int = 4) -> str:
    """Format a number for display.

    * Fractions are shown exactly (``3/2``), integers without a decimal point.
    * Floats are rounded to ``digits`` decimals with trailing zeros removed.
    * Infinity is shown as ``inf``; ``None`` as ``-``.
    """
    if x is None:
        return "-"
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        if x.denominator <= 10**6 and abs(x.numerator) < 10**12:
            return f"{x.numerator}/{x.denominator}"
        return "~" + fmt_num(float(x), digits)   # unwieldy fraction: show a decimal
    if isinstance(x, numbers.Integral):
        return str(int(x))
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    s = f"{x:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", "", "+0"):
        s = "0"
    return s


def fmt_bigm(p, q, digits: int = 4) -> str:
    """Format the Big-M expression ``p*M + q`` compactly, e.g. ``-2M+3`` or ``M``."""
    ps = fmt_num(p, digits)
    qs = fmt_num(q, digits)
    if ps == "0":
        return qs
    if ps == "1":
        m_part = "M"
    elif ps == "-1":
        m_part = "-M"
    elif "/" in ps:  # (5/3)M rather than the ambiguous 5/3M
        m_part = f"-({ps[1:]})M" if ps.startswith("-") else f"({ps})M"
    else:
        m_part = ps + "M"
    if qs == "0":
        return m_part
    if qs.startswith("-"):
        return f"{m_part}-{qs[1:]}"
    return f"{m_part}+{qs}"


def exact_str(value) -> str:
    """Exact text for a number that the parser reads back unchanged: integers as
    they are, terminating decimals in decimal notation (``0.3``), others as ``p/q``."""
    f = to_fraction(value)
    if f.denominator == 1:
        return str(f.numerator)
    d, twos, fives = f.denominator, 0, 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        return f"{f.numerator}/{f.denominator}"
    places = max(twos, fives)
    scaled = f.numerator * (10**places // f.denominator)   # exact: denominator divides 10^places
    text = str(abs(scaled)).rjust(places + 1, "0")
    sign = "-" if scaled < 0 else ""
    return f"{sign}{text[:-places]}.{text[-places:].rstrip('0')}"


def fmt_term(coef, name: str, digits: int = 4, fmt=None) -> str:
    """Format ``|coef| * name`` for a linear expression (sign handled by caller)."""
    s = fmt(abs(coef)) if fmt else fmt_num(abs(coef), digits)
    if s == "1":
        return name
    if "/" in s or name[:1] in "eE":
        return f"{s} {name}"   # "3 e1", not "3e1", which would read as the number 30
    return f"{s}{name}" if name[:1].isalpha() else f"{s} {name}"


def fmt_linear(coeffs, names, digits: int = 4, constant=0, fmt=None) -> str:
    """Format a linear expression such as ``3x1 + 5x2 - x3``.

    ``fmt`` formats one number (default: :func:`fmt_num` with ``digits`` decimals).
    """
    if fmt is None:
        def fmt(v):
            return fmt_num(v, digits)
    parts = []
    for a, name in zip(coeffs, names):
        if fmt(a) == "0":
            continue
        parts.append(("-" if a < 0 else "+", fmt_term(a, name, digits, fmt)))
    if constant is not None and fmt(constant) != "0":
        parts.append(("-" if constant < 0 else "+", fmt(abs(constant))))
    if not parts:
        return "0"
    sign, term = parts[0]
    out = ("-" if sign == "-" else "") + term
    for sign, term in parts[1:]:
        out += f" {sign} {term}"
    return out


def text_table(headers, rows, align=None, indent: str = "  ") -> str:
    """Render a simple ASCII table.

    ``align`` is a string with one character per column: ``l`` (left) or
    ``r`` (right).  Defaults to left for the first column, right otherwise.
    """
    ncol = len(headers)
    if align is None:
        align = "l" + "r" * (ncol - 1)
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(ncol)]

    def fmt_row(row):
        out = []
        for i, c in enumerate(row):
            out.append(c.ljust(widths[i]) if align[i] == "l" else c.rjust(widths[i]))
        return indent + "  ".join(out).rstrip()

    sep = indent + "  ".join("-" * w for w in widths)
    lines = [fmt_row(cells[0]), sep]
    lines.extend(fmt_row(r) for r in cells[1:])
    return "\n".join(lines)


def heading(title: str, char: str = "=", width: int = 72) -> str:
    return f"{char * width}\n {title}\n{char * width}"
