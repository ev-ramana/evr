import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simplex_solver import LinearProgram  # noqa: E402

EXAMPLES = ROOT / "examples"
METHODS = ["two-phase", "big-m", "dual-simplex"]


@pytest.fixture
def wyndor():
    return LinearProgram.from_file(EXAMPLES / "wyndor.lp")


@pytest.fixture
def reddy_mikks():
    return LinearProgram.from_file(EXAMPLES / "reddy_mikks.lp")
