from __future__ import annotations

import pandas as pd

from .config import DAILY_PATH, M5_PATH


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Daily file not found: {DAILY_PATH}")

    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    required_daily = {"date", "open", "high", "low", "close", "volume", "amount"}
    missing_daily = required_daily - set(daily.columns)
    if missing_daily:
        raise ValueError(f"Daily file is missing columns: {sorted(missing_daily)}")

    if M5_PATH.exists():
        m5 = pd.read_csv(M5_PATH)
        if not m5.empty:
            required_m5 = {"date", "time", "open", "high", "low", "close", "volume", "amount"}
            missing_m5 = required_m5 - set(m5.columns)
            if missing_m5:
                raise ValueError(f"5-minute file is missing columns: {sorted(missing_m5)}")
            m5["date"] = pd.to_datetime(m5["date"])
            m5["time"] = m5["time"].astype(str).str.zfill(5)
            m5["datetime"] = pd.to_datetime(m5["date"].dt.strftime("%Y-%m-%d") + " " + m5["time"])
    else:
        m5 = pd.DataFrame(columns=["date", "time", "datetime", "open", "high", "low", "close", "volume", "amount"])

    return daily.sort_values("date").reset_index(drop=True), m5.sort_values(["date", "time"]).reset_index(drop=True)
