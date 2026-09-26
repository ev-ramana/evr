# Ozark Farms diet problem (Taha): minimum-cost feed mix of corn (x1) and
# soybean meal (x2), in lb/day.  Solved with the dual simplex by default.
# Known answer: x1 = 470.59, x2 = 329.41, z = 437.65

min z = 0.3x1 + 0.9x2
subject to
  total_feed:  x1 + x2 >= 800
  protein:     0.21x1 - 0.30x2 <= 0
  fiber:       0.03x1 - 0.01x2 >= 0
