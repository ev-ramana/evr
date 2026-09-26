# Wyndor Glass Co. (Hillier & Lieberman, Introduction to Operations Research)
# x1 = batches of product 1 (glass door), x2 = batches of product 2 (window) per week
# Known answer: x1 = 2, x2 = 6, z = 36; shadow prices 0, 1.5, 1

Maximize Z = 3x1 + 5x2
subject to
  plant1:      x1        <= 4
  plant2:            2x2 <= 12
  plant3:     3x1 + 2x2  <= 18
  x1, x2 >= 0
