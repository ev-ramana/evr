#!/usr/bin/env python3
"""Capacity benchmark: solve random problems of growing size and compare the
optimal objective with scipy's HiGHS solver (if scipy is installed).

    python benchmarks/benchmark.py            # sizes up to 500 x 500
    python benchmarks/benchmark.py --big      # also 1000 x 1000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simplex_solver import LinearProgram, random_lp  # noqa: E402
from simplex_solver.cli import _scipy_objective  # noqa: E402


def diet_like(m, n, seed):
    rng = np.random.default_rng(seed)
    return LinearProgram(c=rng.integers(1, 20, size=n), A=rng.integers(0, 10, size=(m, n)),
                         b=rng.integers(10, 100, size=m), types=">=", sense="min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="include 1000 x 1000")
    args = ap.parse_args()
    sizes = [50, 100, 200, 300, 500] + ([1000] if args.big else [])
    print(f"{'problem':<34}{'method':<22}{'iterations':>11}{'time [s]':>10}{'|obj - scipy|':>15}")
    for size in sizes:
        for label, lp in ((f"max, mixed <=,>=,=  {size}x{size}", random_lp(size, size, seed=1)),
                          (f"min, all >=         {size}x{size}", diet_like(size, size, seed=1))):
            t = time.perf_counter()
            res = lp.solve()
            elapsed = time.perf_counter() - t
            ref, _ = _scipy_objective(lp)
            diff = f"{abs(float(res.objective) - ref):.1e}" if ref is not None else "-"
            print(f"{label:<34}{res.method_key:<22}{res.iterations:>11}{elapsed:>10.2f}{diff:>15}")


if __name__ == "__main__":
    main()
