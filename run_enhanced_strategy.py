from __future__ import annotations

import sys

from scripts.diagnose_ml_underperformance import main as diagnose_main
from scripts.run_full_workflow import main as full_main


if __name__ == "__main__":
    if "--diagnose-only" in sys.argv:
        diagnose_main()
    else:
        full_main()
