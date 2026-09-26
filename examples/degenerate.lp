# Beale's example: the simplex method can cycle on this degenerate problem with
# some tie-breaking rules.  This solver switches to Bland's anti-cycling rule
# automatically, so it always terminates.  Try --bland to force Bland's rule.
# Known answer: x4 = 1, x6 = 1 (others 0), z = 5/4

max z = 3/4 x4 - 20x5 + 1/2 x6 - 6x7
subject to
  1/4 x4 -  8x5 -     x6 + 9x7 <= 0
  1/2 x4 - 12x5 - 1/2 x6 + 3x7 <= 0
                      x6       <= 1
