"""Allow ``python -m simplex_solver problem.lp``."""

import sys

from .cli import main

sys.exit(main())
