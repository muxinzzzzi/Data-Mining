from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import OUTPUT_DIR, PLOT_DIR
from src.strategy.drawdown_event_diagnostics import (
    print_drawdown_event_diagnostics,
    run_drawdown_event_diagnostics,
)


def main() -> None:
    strategy_daily_path = OUTPUT_DIR / "strategy_daily_all.csv"
    if not strategy_daily_path.exists():
        raise FileNotFoundError(f"Missing strategy daily output: {strategy_daily_path}")
    strategy_daily = pd.read_csv(strategy_daily_path)
    report = run_drawdown_event_diagnostics(strategy_daily, OUTPUT_DIR, PLOT_DIR)
    print_drawdown_event_diagnostics(report)


if __name__ == "__main__":
    main()
