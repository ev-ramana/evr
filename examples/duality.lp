# Primal with an equation, to show how the dual is built (Taha, chapter 4)
# The dual is:  min w = 10y1 + 8y2
#               s.t. y1 + 2y2 >= 5,  2y1 - y2 >= 12,  y1 + 3y2 >= 4,  y1 >= 0, y2 free
# Known answer: x1 = 26/5, x2 = 12/5, x3 = 0, z = 274/5;  y1 = 29/5, y2 = -2/5
# Try:  python simplex.py examples/duality.lp --dual --exact

max z = 5x1 + 12x2 + 4x3
subject to
  x1 + 2x2 +  x3 <= 10
 2x1 -  x2 + 3x3  = 8
