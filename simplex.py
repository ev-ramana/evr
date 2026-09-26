#!/usr/bin/env python3
"""Run the simplex solver without installing it:

    python simplex.py examples/wyndor.lp --steps --exact
    python simplex.py --help
"""

import sys

from simplex_solver.cli import main

if __name__ == "__main__":
    sys.exit(main())
