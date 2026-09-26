# Big-M / two-phase example (Taha): '=', '>=' and '<=' constraints
# Try:  python simplex.py examples/bigm.lp --method big-m --steps --exact
# Known answer: x1 = 2/5, x2 = 9/5, z = 17/5

min z = 4x1 + x2
subject to
  3x1 +  x2 = 3
  4x1 + 3x2 >= 6
   x1 + 2x2 <= 4
