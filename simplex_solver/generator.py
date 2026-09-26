"""Random test problems (used by the tests and the capacity benchmark)."""

from __future__ import annotations

import numpy as np

from .model import LinearProgram


def random_lp(m: int, n: int, seed=None, density: float = 1.0, sense: str = "max",
              mixed: bool = True) -> LinearProgram:
    """Generate a random **feasible and bounded** linear program with ``m``
    constraints and ``n`` variables (integer data).

    A random point ``x0 >= 0`` is chosen and the right-hand sides are built so
    that ``x0`` satisfies every constraint.  With ``mixed=True`` the relations are
    a mix of ``<=`` (about 60%), ``>=`` and ``=``; otherwise all are ``<=``.
    Every variable appears in some ``<=`` row with a positive coefficient, so a
    maximisation is bounded.
    """
    if m < 1 or n < 1:
        raise ValueError("need at least one constraint and one variable")
    rng = np.random.default_rng(seed)
    A = rng.integers(1, 10, size=(m, n))
    if density < 1.0:
        A = A * (rng.random((m, n)) < density)
    x0 = rng.integers(0, 10, size=n)
    if mixed:
        types = rng.choice(["<=", ">=", "="], size=m, p=[0.6, 0.25, 0.15]).tolist()
    else:
        types = ["<="] * m
    le_rows = [i for i, t in enumerate(types) if t == "<="]
    if not le_rows:
        types[0] = "<="
        le_rows = [0]
    for j in range(n):
        if not np.any(A[le_rows, j] > 0):
            A[rng.choice(le_rows), j] = rng.integers(1, 10)
    lhs = A @ x0
    b = []
    for i, t in enumerate(types):
        if t == "<=":
            b.append(int(lhs[i] + rng.integers(0, 20)))
        elif t == ">=":
            b.append(int(lhs[i] - rng.integers(0, 20)))
        else:
            b.append(int(lhs[i]))
    c = rng.integers(1, 20, size=n)
    return LinearProgram(c=c, A=A, b=b, types=types, sense=sense,
                         name=f"random {m}x{n}" + (f" (seed {seed})" if seed is not None else ""))
