# Reddy Mikks paint company (Taha, Operations Research: An Introduction)
# x1 = tons of exterior paint, x2 = tons of interior paint per day (profit in $1000)
# Known answer: x1 = 3, x2 = 1.5, z = 21
# Worth of the resources (shadow prices): M1 = 0.75, M2 = 0.5, market = 0, demand = 0

max z = 5x1 + 4x2
subject to
  raw_material_M1:  6x1 + 4x2 <= 24
  raw_material_M2:   x1 + 2x2 <= 6
  market_limit:     -x1 +  x2 <= 1
  demand_limit:             x2 <= 2
