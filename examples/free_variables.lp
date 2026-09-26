# Variables that are unrestricted in sign (free) or non-positive.
# Internally x2 = x2+ - x2- (both >= 0) and x3 = -x3' (x3' >= 0).
# Known answer: x1 = 5, x2 = -1, x3 = -2, z = 20

max z = 4x1 - 2x2 + x3
subject to
  x1 - x2 <= 6
  x1 + x3 <= 3
  x1      <= 5
  x2 free
  x3 <= 0
