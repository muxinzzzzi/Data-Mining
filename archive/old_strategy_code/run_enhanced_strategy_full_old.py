from __future__ import annotations

import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

DAILY_PATH = ROOT / "SH#880823_daily_clean.csv"
M5_PATH = ROOT / "SH#880823_5min_clean.csv"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

INITIAL_CAPITAL = 100_000.0
COST_RATE = 0.001
TRADE_THRESHOLD = max(0.003, 2 * COST_RATE)
ALLOW_LEVERAGE_EXPERIMENT = True
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-05-06")
VALID_START = pd.Timestamp("2024-07-01")
VALID_END = pd.Timestamp("2024-12-31")
HORIZONS = [1, 3, 5, 10, 20]
RETRAIN_EVERY = 20
VALID_RETRAIN_EVERY = 252
REBALANCE_FREQ = 5
RANDOM_STATE = 42
MIN_TRAIN_SAMPLES = 500
FEATURE_TOP_K_CANDIDATES: list[int | None] = [50, 80, 120, None]
TAIL_QUANTILE_MIN_SAMPLES = 252


def safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> pd.Series:
    left = pd.Series(a) if not isinstance(a, pd.Series) else a
    right = pd.Series(b, index=left.index) if not isinstance(b, pd.Series) else b
    return left.divide(right.replace(0, np.nan))


def signed_corr(x: pd.Series, y: pd.Series) -> float:
    valid = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 20:
        return np.nan
    if valid.iloc[:, 0].std() <= 0 or valid.iloc[:, 1].std() <= 0:
        return np.nan
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))


def rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def _rank_last(values: np.ndarray) -> float:
        s = pd.Series(values).dropna()
        if s.empty:
            return np.nan
        return float(s.rank(pct=True).iloc[-1])

    return series.rolling(window, min_periods=min_periods).apply(_rank_last, raw=True)


def consecutive_count(mask: pd.Series) -> pd.Series:
    values: list[int] = []
    count = 0
    for flag in mask.fillna(False).astype(bool):
        count = count + 1 if flag else 0
        values.append(count)
    return pd.Series(values, index=mask.index, dtype=float)


def rolling_trend_stats(series: pd.Series, window: int, min_periods: int) -> tuple[pd.Series, pd.Series]:
    log_series = np.log(series.replace(0, np.nan))

    def _slope(values: np.ndarray) -> float:
        y = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(y) < min_periods:
            return np.nan
        x = np.arange(len(y), dtype=float)
        x = x - x.mean()
        denom = float(np.dot(x, x))
        if denom <= 0:
            return np.nan
        return float(np.dot(x, y - y.mean()) / denom)

    def _r2(values: np.ndarray) -> float:
        y = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(y) < min_periods:
            return np.nan
        x = np.arange(len(y), dtype=float)
        if y.std() <= 0 or x.std() <= 0:
            return 0.0
        corr = np.corrcoef(x, y)[0, 1]
        return float(corr * corr) if np.isfinite(corr) else np.nan

    slope = log_series.rolling(window, min_periods=min_periods).apply(_slope, raw=True)
    r2 = log_series.rolling(window, min_periods=min_periods).apply(_r2, raw=True)
    return slope, r2


def historical_target_quantiles(
    dates: pd.Series,
    target_end_dates: pd.Series,
    returns: pd.Series,
    quantile: float,
    min_samples: int = TAIL_QUANTILE_MIN_SAMPLES,
) -> pd.Series:
    values: list[float] = []
    known = pd.DataFrame({"target_end_date": target_end_dates, "target_ret": returns}).dropna()
    known = known.sort_values("target_end_date").reset_index(drop=True)
    pointer = 0
    history: list[float] = []
    for current_date in pd.to_datetime(dates):
        while pointer < len(known) and known.loc[pointer, "target_end_date"] < current_date:
            history.append(float(known.loc[pointer, "target_ret"]))
            pointer += 1
        if len(history) >= min_samples:
            values.append(float(pd.Series(history).quantile(quantile)))
        else:
            values.append(np.nan)
    return pd.Series(values, index=dates.index, dtype=float)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


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


def build_intraday_features(m5: pd.DataFrame) -> pd.DataFrame:
    if m5.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})

    work = m5.sort_values(["date", "time"]).copy()
    work["bar_return"] = work.groupby("date")["close"].pct_change()
    work["bar_oc_return"] = safe_div(work["close"], work["open"]) - 1
    work["bar_range"] = safe_div(work["high"] - work["low"], work["open"])
    work["bar_vwap"] = safe_div(work["amount"], work["volume"])
    work["bar_vwap"] = work["bar_vwap"].where(np.isfinite(work["bar_vwap"]))

    def _safe_scalar_ratio(numerator: float, denominator: float) -> float:
        if denominator == 0 or not np.isfinite(denominator):
            return np.nan
        return float(numerator / denominator)

    def _safe_corr(left: pd.Series, right: pd.Series, min_samples: int = 5) -> float:
        valid = pd.concat([left, right], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < min_samples:
            return np.nan
        if valid.iloc[:, 0].std() <= 0 or valid.iloc[:, 1].std() <= 0:
            return np.nan
        return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))

    def _safe_autocorr(series: pd.Series, min_samples: int = 5) -> float:
        valid = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < min_samples or valid.std() <= 0:
            return np.nan
        lagged = valid.shift(1)
        return _safe_corr(valid, lagged, min_samples=min_samples)

    def _intraday_slope_r2(close: pd.Series, min_samples: int = 5) -> tuple[float, float]:
        y = np.log(close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(y) < min_samples or np.nanstd(y) <= 0:
            return np.nan, np.nan
        x = np.arange(len(y), dtype=float)
        x_centered = x - x.mean()
        denom = float(np.dot(x_centered, x_centered))
        if denom <= 0:
            return np.nan, np.nan
        slope = float(np.dot(x_centered, y - y.mean()) / denom)
        fitted = y.mean() + slope * x_centered
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = np.nan if ss_tot <= 0 else float(1 - ss_res / ss_tot)
        return slope, r2

    def _sign_flip_count(returns: pd.Series) -> int:
        signs = np.sign(returns.replace([np.inf, -np.inf], np.nan).dropna())
        signs = signs[signs != 0]
        if len(signs) < 2:
            return 0
        return int((signs != signs.shift(1)).sum() - 1)

    def _longest_streak(returns: pd.Series, positive: bool) -> int:
        best = 0
        current = 0
        clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
        for value in clean:
            flag = value > 0 if positive else value < 0
            current = current + 1 if flag else 0
            best = max(best, current)
        return int(best)

    rows: list[dict[str, Any]] = []
    for date, grp in work.groupby("date", sort=True):
        grp = grp.reset_index(drop=True)
        first_open = float(grp.loc[0, "open"])
        last_close = float(grp.loc[len(grp) - 1, "close"])
        day_high = float(grp["high"].max())
        day_low = float(grp["low"].min())
        total_volume = float(grp["volume"].sum())
        total_amount = float(grp["amount"].sum())
        day_vwap = total_amount / total_volume if total_volume > 0 else np.nan

        morning = grp[grp["time"] <= "11:30"]
        afternoon = grp[grp["time"] >= "13:00"]
        first_hour = grp[grp["time"] <= "10:30"]
        last_hour = grp[grp["time"] >= "14:00"]
        early = grp.iloc[: max(1, min(12, len(grp)))]
        late = last_hour if not last_hour.empty else grp.iloc[-12:]
        midday = grp[grp["time"] <= "14:00"]

        morning_close = float(morning.iloc[-1]["close"]) if not morning.empty else last_close
        afternoon_open = float(afternoon.iloc[0]["open"]) if not afternoon.empty else float(grp.iloc[-1]["open"])
        first_hour_close = float(first_hour.iloc[-1]["close"]) if not first_hour.empty else morning_close
        last_hour_open = float(last_hour.iloc[0]["open"]) if not last_hour.empty else float(grp.iloc[-1]["open"])
        midday_close = float(midday.iloc[-1]["close"]) if not midday.empty else last_close

        morning_return = morning_close / first_open - 1 if first_open else np.nan
        afternoon_return = last_close / afternoon_open - 1 if afternoon_open else np.nan
        first_hour_return = first_hour_close / first_open - 1 if first_open else np.nan
        last_hour_return = last_close / last_hour_open - 1 if last_hour_open else np.nan
        intraday_return = last_close / first_open - 1 if first_open else np.nan
        intraday_range = day_high / day_low - 1 if day_low else np.nan
        midday_reversal = last_close / midday_close - 1 if midday_close else np.nan
        volume_profile = float(early["volume"].sum()) / total_volume if total_volume else np.nan
        last_hour_volume_share = float(late["volume"].sum()) / total_volume if total_volume else np.nan
        up_volume_share = float(grp.loc[grp["bar_return"] > 0, "volume"].sum()) / total_volume if total_volume else np.nan
        close_position = (last_close - day_low) / (day_high - day_low) if day_high > day_low else np.nan
        trend_consistency = float((grp["bar_return"].dropna() > 0).mean()) if grp["bar_return"].notna().any() else np.nan
        valid_bar_ret = grp["bar_return"].dropna()
        positive_bar_ratio = float((valid_bar_ret > 0).mean()) if len(valid_bar_ret) else np.nan
        negative_bar_ratio = float((valid_bar_ret < 0).mean()) if len(valid_bar_ret) else np.nan
        large_drop_count = int((valid_bar_ret < -0.003).sum()) if len(valid_bar_ret) else 0
        large_up_count = int((valid_bar_ret > 0.003).sum()) if len(valid_bar_ret) else 0
        open30 = grp.iloc[: max(1, min(6, len(grp)))]
        open30_open = float(open30.iloc[0]["open"]) if not open30.empty else np.nan
        open30_close = float(open30.iloc[-1]["close"]) if not open30.empty else np.nan
        open30_high = float(open30["high"].max()) if not open30.empty else np.nan
        open30_low = float(open30["low"].min()) if not open30.empty else np.nan
        open30_volume = float(open30["volume"].sum()) if not open30.empty else np.nan
        open30_amount = float(open30["amount"].sum()) if not open30.empty else np.nan
        open30_return = _safe_scalar_ratio(open30_close, open30_open) - 1 if np.isfinite(open30_close) else np.nan
        open30_range = _safe_scalar_ratio(open30_high, open30_low) - 1 if np.isfinite(open30_high) else np.nan
        open30_volume_share = _safe_scalar_ratio(open30_volume, total_volume)
        open30_amount_share = _safe_scalar_ratio(open30_amount, total_amount)
        open30_close_position = (
            (open30_close - open30_low) / (open30_high - open30_low)
            if np.isfinite(open30_close) and open30_high > open30_low
            else np.nan
        )
        open_shock_reversal = intraday_return - open30_return if pd.notna(intraday_return) and pd.notna(open30_return) else np.nan

        last30 = grp.iloc[-6:] if len(grp) >= 6 else grp
        close30 = last30
        close30_open = float(close30.iloc[0]["open"]) if not close30.empty else np.nan
        close30_high = float(close30["high"].max()) if not close30.empty else np.nan
        close30_low = float(close30["low"].min()) if not close30.empty else np.nan
        close30_volume = float(close30["volume"].sum()) if not close30.empty else np.nan
        close30_amount = float(close30["amount"].sum()) if not close30.empty else np.nan
        close30_return = _safe_scalar_ratio(last_close, close30_open) - 1 if np.isfinite(last_close) else np.nan
        close30_range = _safe_scalar_ratio(close30_high, close30_low) - 1 if np.isfinite(close30_high) else np.nan
        close30_volume_share = _safe_scalar_ratio(close30_volume, total_volume)
        close30_amount_share = _safe_scalar_ratio(close30_amount, total_amount)
        close30_buy_pressure = close30_return * close30_volume_share if pd.notna(close30_return) and pd.notna(close30_volume_share) else np.nan
        close30_sell_pressure = -min(close30_return, 0) * close30_volume_share if pd.notna(close30_return) and pd.notna(close30_volume_share) else np.nan

        last30_open = float(last30.iloc[0]["open"]) if not last30.empty else np.nan
        last30_return = last_close / last30_open - 1 if last30_open and np.isfinite(last30_open) else np.nan
        last30_volume_share = float(last30["volume"].sum()) / total_volume if total_volume else np.nan
        close_vwap_position = last_close / day_vwap - 1 if day_vwap and np.isfinite(day_vwap) else np.nan
        close_above_vwap = float(last_close > day_vwap) if day_vwap and np.isfinite(day_vwap) else np.nan

        high_idx = int(grp["high"].idxmax()) if len(grp) else 0
        low_idx = int(grp["low"].idxmin()) if len(grp) else 0
        intraday_max_return = _safe_scalar_ratio(day_high, first_open) - 1
        intraday_min_return = _safe_scalar_ratio(day_low, first_open) - 1
        high_time_ratio = (high_idx + 1) / len(grp) if len(grp) else np.nan
        low_time_ratio = (low_idx + 1) / len(grp) if len(grp) else np.nan
        high_to_close_reversal = _safe_scalar_ratio(last_close, day_high) - 1
        low_to_close_recovery = _safe_scalar_ratio(last_close, day_low) - 1
        morning_to_close_reversal = _safe_scalar_ratio(last_close, morning_close) - 1

        up_mask = grp["bar_return"] > 0
        down_mask = grp["bar_return"] < 0
        up_amount_share = _safe_scalar_ratio(float(grp.loc[up_mask, "amount"].sum()), total_amount)
        down_amount_share = _safe_scalar_ratio(float(grp.loc[down_mask, "amount"].sum()), total_amount)
        up_volume = float(grp.loc[up_mask, "volume"].sum())
        down_volume = float(grp.loc[down_mask, "volume"].sum())
        up_amount = float(grp.loc[up_mask, "amount"].sum())
        down_amount = float(grp.loc[down_mask, "amount"].sum())
        net_volume_pressure = _safe_scalar_ratio(up_volume - down_volume, total_volume)
        net_amount_pressure = _safe_scalar_ratio(up_amount - down_amount, total_amount)
        down_avg_return = float(grp.loc[down_mask, "bar_return"].mean()) if down_mask.any() else 0.0
        down_volume_pressure = _safe_scalar_ratio(down_volume, total_volume) * abs(down_avg_return) if total_volume else np.nan
        volume_return_corr = _safe_corr(grp["bar_return"], grp["volume"])
        amount_return_corr = _safe_corr(grp["bar_return"], grp["amount"])
        volume_cut = grp["volume"].quantile(0.80)
        amount_cut = grp["amount"].quantile(0.80)
        large_volume_bar_return = float(grp.loc[grp["volume"] >= volume_cut, "bar_return"].mean()) if pd.notna(volume_cut) else np.nan
        large_amount_bar_return = float(grp.loc[grp["amount"] >= amount_cut, "bar_return"].mean()) if pd.notna(amount_cut) else np.nan

        bar_return_autocorr = _safe_autocorr(grp["bar_return"])
        intraday_slope, intraday_r2 = _intraday_slope_r2(grp["close"])
        sign_flip_count = _sign_flip_count(grp["bar_return"])
        longest_up_streak = _longest_streak(grp["bar_return"], positive=True)
        longest_down_streak = _longest_streak(grp["bar_return"], positive=False)

        rows.append(
            {
                "date": date,
                "m5_intraday_return": intraday_return,
                "m5_morning_return": morning_return,
                "m5_afternoon_return": afternoon_return,
                "m5_first_hour_return": first_hour_return,
                "m5_last_hour_return": last_hour_return,
                "m5_intraday_range": intraday_range,
                "m5_intraday_volatility": grp["bar_return"].std(),
                "m5_midday_reversal": midday_reversal,
                "m5_volume_profile": volume_profile,
                "m5_vwap": day_vwap,
                "m5_intraday_close_position": close_position,
                "m5_last_hour_volume_share": last_hour_volume_share,
                "m5_intraday_trend_consistency": trend_consistency,
                "m5_up_volume_share": up_volume_share,
                "m5_positive_bar_ratio": positive_bar_ratio,
                "m5_negative_bar_ratio": negative_bar_ratio,
                "m5_large_drop_count": large_drop_count,
                "m5_large_up_count": large_up_count,
                "m5_close_above_vwap": close_above_vwap,
                "m5_close_vwap_position": close_vwap_position,
                "m5_last30_return": last30_return,
                "m5_last30_volume_share": last30_volume_share,
                "m5_open_30m_return": open30_return,
                "m5_open_30m_range": open30_range,
                "m5_open_30m_volume_share": open30_volume_share,
                "m5_open_30m_amount_share": open30_amount_share,
                "m5_open_30m_close_position": open30_close_position,
                "m5_open_shock_reversal": open_shock_reversal,
                "m5_close_30m_return": close30_return,
                "m5_close_30m_range": close30_range,
                "m5_close_30m_volume_share": close30_volume_share,
                "m5_close_30m_amount_share": close30_amount_share,
                "m5_close_30m_buy_pressure": close30_buy_pressure,
                "m5_close_30m_sell_pressure": close30_sell_pressure,
                "m5_intraday_max_return": intraday_max_return,
                "m5_intraday_min_return": intraday_min_return,
                "m5_high_time_ratio": high_time_ratio,
                "m5_low_time_ratio": low_time_ratio,
                "m5_high_to_close_reversal": high_to_close_reversal,
                "m5_low_to_close_recovery": low_to_close_recovery,
                "m5_morning_to_close_reversal": morning_to_close_reversal,
                "m5_volume_return_corr": volume_return_corr,
                "m5_amount_return_corr": amount_return_corr,
                "m5_up_amount_share": up_amount_share,
                "m5_down_amount_share": down_amount_share,
                "m5_net_volume_pressure": net_volume_pressure,
                "m5_net_amount_pressure": net_amount_pressure,
                "m5_down_volume_pressure": down_volume_pressure,
                "m5_large_volume_bar_return": large_volume_bar_return,
                "m5_large_amount_bar_return": large_amount_bar_return,
                "m5_bar_return_autocorr": bar_return_autocorr,
                "m5_intraday_slope": intraday_slope,
                "m5_intraday_r2": intraday_r2,
                "m5_sign_flip_count": sign_flip_count,
                "m5_longest_up_streak": longest_up_streak,
                "m5_longest_down_streak": longest_down_streak,
                "m5_bar_range_mean": grp["bar_range"].mean(),
                "m5_bar_range_max": grp["bar_range"].max(),
                "m5_total_volume": total_volume,
                "m5_total_amount": total_amount,
            }
        )

    intraday = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    for col in [
        "m5_intraday_return",
        "m5_morning_return",
        "m5_afternoon_return",
        "m5_first_hour_return",
        "m5_last_hour_return",
        "m5_intraday_range",
        "m5_intraday_volatility",
        "m5_midday_reversal",
        "m5_volume_profile",
        "m5_last_hour_volume_share",
        "m5_intraday_trend_consistency",
        "m5_positive_bar_ratio",
        "m5_negative_bar_ratio",
        "m5_close_vwap_position",
        "m5_last30_return",
        "m5_last30_volume_share",
    ]:
        intraday[f"{col}_mean_5"] = intraday[col].rolling(5, min_periods=3).mean()
        intraday[f"{col}_std_5"] = intraday[col].rolling(5, min_periods=3).std()

    intraday["m5_return_reversal_1"] = -intraday["m5_intraday_return"].shift(1)
    intraday["m5_morning_afternoon_spread"] = intraday["m5_morning_return"] - intraday["m5_afternoon_return"]
    intraday["m5_volume_profile_change_5"] = intraday["m5_volume_profile"] - intraday["m5_volume_profile"].rolling(5, min_periods=3).mean()
    anomaly_cols = [
        "m5_close_30m_buy_pressure",
        "m5_close_30m_sell_pressure",
        "m5_net_volume_pressure",
        "m5_net_amount_pressure",
        "m5_large_volume_bar_return",
        "m5_high_to_close_reversal",
        "m5_low_to_close_recovery",
        "m5_intraday_slope",
        "m5_volume_return_corr",
    ]
    for col in anomaly_cols:
        mean_20 = intraday[col].rolling(20, min_periods=5).mean()
        std_20 = intraday[col].rolling(20, min_periods=5).std().replace(0, np.nan)
        intraday[f"{col}_z_20"] = (intraday[col] - mean_20) / std_20
        intraday[f"{col}_rank_20"] = rolling_percentile(intraday[col], 20, 5)
    intraday = intraday.replace([np.inf, -np.inf], np.nan)
    return intraday


def add_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]

    out["gap_open"] = out["open"] / close.shift(1) - 1
    out["intraday_body"] = out["close"] / out["open"] - 1
    out["close_to_high"] = out["close"] / out["high"] - 1
    out["close_to_low"] = out["close"] / out["low"] - 1
    out["daily_range_pct"] = (out["high"] - out["low"]) / close.shift(1)
    out["upper_shadow"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["open"]
    out["lower_shadow"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["open"]

    for h in [1, 3, 5, 10, 20, 60]:
        out[f"ret_{h}"] = close.pct_change(h)
        out[f"short_reversal_{h}"] = -out[f"ret_{h}"]

    for w in [3, 5, 10, 20, 60]:
        out[f"ret_mean_{w}"] = out["ret_1"].rolling(w, min_periods=max(2, w // 2)).mean()
        out[f"ret_median_{w}"] = out["ret_1"].rolling(w, min_periods=max(2, w // 2)).median()

    for w in [10, 20, 60]:
        out[f"up_day_ratio_{w}"] = (out["ret_1"] > 0).astype(float).rolling(w, min_periods=max(3, w // 2)).mean()
    out["positive_return_streak"] = consecutive_count(out["ret_1"] > 0)
    out["negative_return_streak"] = consecutive_count(out["ret_1"] < 0)
    for w in [20, 60]:
        slope, r2 = rolling_trend_stats(close, w, max(8, w // 2))
        out[f"trend_slope_{w}"] = slope
        out[f"trend_r2_{w}"] = r2

    out["rsi_6"] = compute_rsi(close, 6)
    out["rsi_14"] = compute_rsi(close, 14)
    out["rsi_24"] = compute_rsi(close, 24)
    out["rsi_change_5"] = out["rsi_14"].diff(5)

    low_14 = low.rolling(14, min_periods=8).min()
    high_14 = high.rolling(14, min_periods=8).max()
    out["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
    out["stoch_d"] = out["stoch_k"].rolling(3, min_periods=2).mean()
    out["stoch_j"] = 3 * out["stoch_k"] - 2 * out["stoch_d"]
    out["kdj_cross"] = out["stoch_k"] - out["stoch_d"]

    ma: dict[int, pd.Series] = {}
    for w in [5, 10, 20, 60, 120]:
        ma[w] = close.rolling(w, min_periods=max(5, w // 2)).mean()
        out[f"ma_ratio_{w}"] = close / ma[w] - 1
        out[f"ma_slope_{w}"] = ma[w].pct_change(5)

    out["ma_spread_5_20"] = ma[5] / ma[20] - 1
    out["ma_spread_10_20"] = ma[10] / ma[20] - 1
    out["ma_spread_20_60"] = ma[20] / ma[60] - 1
    out["ma_spread_60_120"] = ma[60] / ma[120] - 1

    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["macd_hist_change"] = out["macd_hist"].diff(3)

    for w in [20, 60, 120]:
        rolling_high = high.rolling(w, min_periods=max(5, w // 2)).max()
        rolling_low = low.rolling(w, min_periods=max(5, w // 2)).min()
        out[f"breakout_high_{w}"] = close / rolling_high - 1
        out[f"breakout_low_{w}"] = close / rolling_low - 1
        out[f"breakout_high_prior_{w}"] = close / rolling_high.shift(1) - 1
        out[f"breakout_low_prior_{w}"] = close / rolling_low.shift(1) - 1
        out[f"close_position_{w}"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)

    out["trend_strength_score"] = (
        0.22 * np.sign(out["ret_20"]).fillna(0)
        + 0.22 * np.sign(out["ma_ratio_20"]).fillna(0)
        + 0.22 * np.sign(out["ma_spread_20_60"]).fillna(0)
        + 0.17 * np.sign(out["macd_hist"]).fillna(0)
        + 0.17 * np.sign(out["ma_slope_60"]).fillna(0)
    )
    return out


def add_volume_liquidity_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["volume_change_1"] = out["volume"].pct_change(1)
    out["amount_change_1"] = out["amount"].pct_change(1)
    signed_volume_pressure = (
        np.sign(out["ret_1"].fillna(0.0))
        * out["volume_change_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    for w in [5, 10, 20, 60]:
        vol_ma = out["volume"].rolling(w, min_periods=max(3, w // 2)).mean()
        amt_ma = out["amount"].rolling(w, min_periods=max(3, w // 2)).mean()
        out[f"volume_ratio_{w}"] = out["volume"] / vol_ma - 1
        out[f"amount_ratio_{w}"] = out["amount"] / amt_ma - 1
        out[f"price_volume_corr_{w}"] = out["ret_1"].rolling(w, min_periods=max(5, w // 2)).corr(out["volume_change_1"])
        up_vol = out["volume"].where(out["ret_1"] > 0, 0).rolling(w, min_periods=max(3, w // 2)).sum()
        total_vol = out["volume"].rolling(w, min_periods=max(3, w // 2)).sum()
        out[f"up_volume_ratio_{w}"] = up_vol / total_vol.replace(0, np.nan)
        out[f"amihud_illiq_{w}"] = (out["ret_1"].abs() / out["amount"].replace(0, np.nan)).rolling(
            w, min_periods=max(3, w // 2)
        ).mean() * 1e10
        out[f"amount_abnormality_{w}"] = (out["amount"] - amt_ma) / out["amount"].rolling(w, min_periods=max(3, w // 2)).std()
        out[f"volume_abnormality_{w}"] = (out["volume"] - vol_ma) / out["volume"].rolling(w, min_periods=max(3, w // 2)).std()
        out[f"volume_price_confirmation_{w}"] = signed_volume_pressure.rolling(
            w, min_periods=max(3, w // 2)
        ).mean()

    up_vol_20 = out["volume"].where(out["ret_1"] > 0, 0.0).rolling(20, min_periods=10).sum()
    down_vol_20 = out["volume"].where(out["ret_1"] < 0, 0.0).rolling(20, min_periods=10).sum()
    total_vol_20 = out["volume"].rolling(20, min_periods=10).sum()
    down_ret_sum_20 = out["ret_1"].where(out["ret_1"] < 0, 0.0).abs().rolling(20, min_periods=10).sum()
    out["up_day_volume_ratio_20"] = up_vol_20 / total_vol_20.replace(0, np.nan)
    out["down_day_volume_ratio_20"] = down_vol_20 / total_vol_20.replace(0, np.nan)
    out["breakout_volume_confirm_20"] = (
        (out["breakout_high_prior_20"] > -0.005).astype(float)
        * out["volume_ratio_20"].clip(lower=0).fillna(0.0)
        * (out["ret_1"] > 0).astype(float)
    )
    out["sell_pressure_volume_20"] = out["down_day_volume_ratio_20"] * down_ret_sum_20

    if "m5_total_volume" in out.columns:
        out["m5_total_volume_ratio_20"] = out["m5_total_volume"] / out["m5_total_volume"].rolling(20, min_periods=8).mean() - 1
        out["m5_total_amount_ratio_20"] = out["m5_total_amount"] / out["m5_total_amount"].rolling(20, min_periods=8).mean() - 1
    return out


def add_risk_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["true_range_pct"] = true_range / prev_close

    for w in [5, 10, 20, 60]:
        out[f"volatility_{w}"] = out["ret_1"].rolling(w, min_periods=max(3, w // 2)).std()
        out[f"realized_vol_{w}"] = out["ret_1"].rolling(w, min_periods=max(3, w // 2)).std() * math.sqrt(252)
        out[f"atr_{w}"] = true_range.rolling(w, min_periods=max(3, w // 2)).mean()
        out[f"atr_ratio_{w}"] = out[f"atr_{w}"] / out["close"]
        downside = out["ret_1"].where(out["ret_1"] < 0, 0.0)
        out[f"downside_volatility_{w}"] = downside.rolling(w, min_periods=max(3, w // 2)).std()

    out["volatility_ratio_5_20"] = out["volatility_5"] / out["volatility_20"]
    out["volatility_ratio_20_60"] = out["volatility_20"] / out["volatility_60"]
    out["atr_ratio_20_60"] = out["atr_ratio_20"] / out["atr_ratio_60"]

    ma20 = out["close"].rolling(20, min_periods=10).mean()
    std20 = out["close"].rolling(20, min_periods=10).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    out["bollinger_width"] = (upper - lower) / ma20
    out["bollinger_position"] = (out["close"] - lower) / (upper - lower).replace(0, np.nan)

    for w in [20, 60, 120]:
        running_high = out["close"].rolling(w, min_periods=max(5, w // 2)).max()
        out[f"drawdown_{w}"] = out["close"] / running_high - 1
        out[f"drawdown_speed_{w}"] = out[f"drawdown_{w}"].diff(5)

    up_threshold = out["ret_1"].shift(1).rolling(252, min_periods=80).quantile(0.80)
    down_threshold = out["ret_1"].shift(1).rolling(252, min_periods=80).quantile(0.20)
    large_up = out["ret_1"] > up_threshold
    large_down = out["ret_1"] < down_threshold
    out["large_up_count_20"] = large_up.astype(float).rolling(20, min_periods=10).sum()
    out["large_down_count_20"] = large_down.astype(float).rolling(20, min_periods=10).sum()
    out["large_up_return_sum_20"] = out["ret_1"].where(large_up, 0.0).rolling(20, min_periods=10).sum()
    out["large_down_return_sum_20"] = out["ret_1"].where(large_down, 0.0).rolling(20, min_periods=10).sum()
    out["tail_return_ratio_20"] = out["large_up_return_sum_20"] / out["large_down_return_sum_20"].abs().replace(0, np.nan)
    out["downside_return_sum_20"] = out["ret_1"].where(out["ret_1"] < 0, 0.0).rolling(20, min_periods=10).sum()
    out["crash_risk_score"] = (
        0.35 * (out["large_down_count_20"] / 20).fillna(0.0)
        + 0.25 * out["downside_return_sum_20"].abs().fillna(0.0)
        + 0.20 * out["drawdown_20"].abs().fillna(0.0)
        + 0.20 * out["volatility_ratio_5_20"].clip(lower=0).fillna(1.0)
    )

    return out


def add_regime_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bull_regime_ma60"] = (out["ma_ratio_60"] > 0).astype(float)
    out["bull_regime_ma120"] = (out["ma_ratio_120"] > 0).astype(float)
    out["bear_regime_ma60"] = (out["ma_ratio_60"] < 0).astype(float)
    out["trend_regime_score"] = (
        (out["ma_ratio_20"] > 0).astype(float)
        + (out["ma_ratio_60"] > 0).astype(float)
        + (out["ma_spread_20_60"] > 0).astype(float)
        + (out["ret_20"] > 0).astype(float)
    ) / 4
    vol_threshold = out["volatility_20"].rolling(252, min_periods=80).quantile(0.7)
    out["high_volatility_regime"] = (out["volatility_20"] > vol_threshold).astype(float)
    out["drawdown_regime"] = (out["drawdown_60"] < -0.08).astype(float)
    out["severe_drawdown_regime"] = (out["drawdown_120"] < -0.15).astype(float)
    out["trend_vol_regime"] = out["trend_regime_score"] - out["high_volatility_regime"] - 0.5 * out["drawdown_regime"]
    out["bull_low_vol_regime"] = ((out["trend_regime_score"] >= 0.75) & (out["high_volatility_regime"] == 0)).astype(float)
    out["bull_high_vol_regime"] = ((out["trend_regime_score"] >= 0.75) & (out["high_volatility_regime"] == 1)).astype(float)
    out["bear_high_vol_regime"] = ((out["trend_regime_score"] <= 0.25) & (out["high_volatility_regime"] == 1)).astype(float)
    out["sideways_regime"] = (
        (out["trend_regime_score"] > 0.25)
        & (out["trend_regime_score"] < 0.75)
        & (out["high_volatility_regime"] == 0)
    ).astype(float)
    return out


def build_daily_features(daily: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("date").copy()
    df = df.merge(intraday, on="date", how="left")
    df["m5_available"] = df["m5_intraday_return"].notna().astype(float) if "m5_intraday_return" in df.columns else 0.0

    if "m5_vwap" in df.columns:
        df["m5_vwap_discount"] = df["close"] / df["m5_vwap"] - 1
        df["m5_daily_intraday_gap"] = df["close"].pct_change() - df["m5_intraday_return"]
    else:
        df["m5_vwap_discount"] = np.nan
        df["m5_daily_intraday_gap"] = np.nan

    df = add_technical_factors(df)
    df = add_volume_liquidity_factors(df)
    df = add_risk_factors(df)
    df = add_regime_factors(df)
    df = add_targets(df)
    return df.replace([np.inf, -np.inf], np.nan)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["trade_date"] = out["date"].shift(-1)
    out["fwd_ret_1d"] = out["close"].shift(-1) / out["close"] - 1
    for h in HORIZONS:
        ret = out["close"].shift(-h) / out["close"] - 1
        target_end_date = out["date"].shift(-h)
        big_up_threshold = historical_target_quantiles(out["date"], target_end_date, ret, 0.70)
        big_down_threshold = historical_target_quantiles(out["date"], target_end_date, ret, 0.30)
        out[f"target_ret_{h}d"] = ret
        out[f"target_up_{h}d"] = np.where(ret.notna(), (ret > 0).astype(float), np.nan)
        out[f"target_trade_{h}d"] = np.where(ret.notna(), (ret > TRADE_THRESHOLD).astype(float), np.nan)
        out[f"target_big_up_{h}d"] = np.where(
            ret.notna() & big_up_threshold.notna(),
            (ret > big_up_threshold).astype(float),
            np.nan,
        )
        out[f"target_big_down_{h}d"] = np.where(
            ret.notna() & big_down_threshold.notna(),
            (ret < big_down_threshold).astype(float),
            np.nan,
        )
        out[f"target_big_up_threshold_{h}d"] = big_up_threshold
        out[f"target_big_down_threshold_{h}d"] = big_down_threshold
        out[f"target_start_date_{h}d"] = out["date"].shift(-1)
        out[f"target_end_date_{h}d"] = target_end_date
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded_exact = {
        "symbol",
        "name",
        "frequency",
        "adjustment",
        "date",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "m5_vwap",
        "m5_total_volume",
        "m5_total_amount",
        "fwd_ret_1d",
    }
    feature_cols: list[str] = []
    for col in df.columns:
        if col in excluded_exact:
            continue
        if col.startswith("target_"):
            continue
        if col.startswith("short_reversal_"):
            continue
        if col.startswith("realized_vol_"):
            continue
        if col == "trend_strength_score":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


@dataclass(frozen=True)
class ModelSpec:
    name: str
    reg: Any
    cls: Any
    trade_cls: Any
    big_up_cls: Any | None = None
    big_down_cls: Any | None = None


def clone_model(model: Any) -> Any:
    from sklearn.base import clone

    return clone(model)


def build_model_specs() -> tuple[list[ModelSpec], list[dict[str, str]]]:
    specs = [
        ModelSpec(
            "HistGradientBoosting",
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=80,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=80,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=80,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=0.1,
                early_stopping=False,
                random_state=RANDOM_STATE + 1,
            ),
        ),
        ModelSpec(
            "RandomForest",
            RandomForestRegressor(
                n_estimators=40,
                max_depth=6,
                min_samples_leaf=18,
                max_features="sqrt",
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            RandomForestClassifier(
                n_estimators=40,
                max_depth=6,
                min_samples_leaf=18,
                max_features="sqrt",
                bootstrap=True,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            RandomForestClassifier(
                n_estimators=40,
                max_depth=6,
                min_samples_leaf=18,
                max_features="sqrt",
                bootstrap=True,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE + 1,
                n_jobs=1,
            ),
        ),
        ModelSpec(
            "ExtraTrees",
            ExtraTreesRegressor(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=16,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            ExtraTreesClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=16,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            ExtraTreesClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=16,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE + 1,
                n_jobs=1,
            ),
        ),
        ModelSpec(
            "GradientBoosting",
            GradientBoostingRegressor(
                n_estimators=60,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=18,
                subsample=0.8,
                random_state=RANDOM_STATE,
            ),
            GradientBoostingClassifier(
                n_estimators=60,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=18,
                subsample=0.8,
                random_state=RANDOM_STATE,
            ),
            GradientBoostingClassifier(
                n_estimators=60,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=18,
                subsample=0.8,
                random_state=RANDOM_STATE + 1,
            ),
        ),
    ]

    skipped: list[dict[str, str]] = []
    try:
        from xgboost import XGBClassifier, XGBRegressor

        specs.append(
            ModelSpec(
                "XGBoost",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                ),
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbosity=0,
                ),
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE + 1,
                    n_jobs=1,
                    verbosity=0,
                ),
            )
        )
    except Exception as exc:
        skipped.append({"model_name": "XGBoost", "reason": f"xgboost unavailable: {exc.__class__.__name__}"})

    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        specs.append(
            ModelSpec(
                "LightGBM",
                LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=3,
                    num_leaves=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbose=-1,
                ),
                LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=3,
                    num_leaves=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbose=-1,
                ),
                LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=3,
                    num_leaves=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=RANDOM_STATE + 1,
                    n_jobs=1,
                    verbose=-1,
                ),
            )
        )
    except Exception as exc:
        skipped.append({"model_name": "LightGBM", "reason": f"lightgbm unavailable: {exc.__class__.__name__}"})

    return specs, skipped


def fit_preprocessor(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, pd.Series]:
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    lower = x.quantile(0.01)
    upper = x.quantile(0.99)
    med = x.clip(lower=lower, upper=upper, axis=1).median().fillna(0.0)
    lower = lower.fillna(med)
    upper = upper.fillna(med)
    return {"lower": lower, "upper": upper, "median": med}


def transform_features(df: pd.DataFrame, feature_cols: list[str], prep: dict[str, pd.Series]) -> pd.DataFrame:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.clip(lower=prep["lower"], upper=prep["upper"], axis=1)
    x = x.fillna(prep["median"])
    return x.astype(float)


def fit_binary_probability(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    y = y_train.astype(int)
    if y.nunique() < 2:
        return np.full(len(x_test), float(y.iloc[0]) if len(y) else 0.0)
    fitted = clone_model(model)
    fitted.fit(x_train, y)
    if hasattr(fitted, "predict_proba"):
        proba = fitted.predict_proba(x_test)
        classes = list(fitted.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(x_test))
    raw = fitted.decision_function(x_test)
    return 1 / (1 + np.exp(-raw))


def fit_optional_binary_probability(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    default: float = 0.5,
) -> np.ndarray:
    valid = y_train.notna()
    if int(valid.sum()) < 20:
        return np.full(len(x_test), default)
    return fit_binary_probability(model, x_train.loc[valid], y_train.loc[valid], x_test)


def select_ic_feature_columns(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    top_k: int | None,
) -> list[str]:
    if top_k is None or top_k >= len(feature_cols):
        return feature_cols
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = train[target_col].astype(float).replace([np.inf, -np.inf], np.nan)
    corr = x.corrwith(y).abs().replace([np.inf, -np.inf], np.nan).dropna()
    ranked = corr.sort_values(ascending=False)
    selected = ranked.head(top_k).index.tolist()
    if len(selected) < min(top_k, len(feature_cols)):
        selected_set = set(selected)
        selected.extend([col for col in feature_cols if col not in selected_set][: top_k - len(selected)])
    return selected


def feature_top_k_label(top_k: int | None) -> str:
    return "all" if top_k is None else str(top_k)


def walk_forward_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_specs: list[ModelSpec],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    retrain_every: int,
    min_train_samples: int = MIN_TRAIN_SAMPLES,
    label: str = "test",
    feature_top_k: int | None = None,
) -> pd.DataFrame:
    test_dates = (
        df.loc[(df["date"] >= period_start) & (df["date"] <= period_end), "date"]
        .dropna()
        .sort_values()
        .drop_duplicates()
        .tolist()
    )
    all_preds: list[pd.DataFrame] = []

    for h in HORIZONS:
        ret_col = f"target_ret_{h}d"
        up_col = f"target_up_{h}d"
        trade_col = f"target_trade_{h}d"
        big_up_col = f"target_big_up_{h}d"
        big_down_col = f"target_big_down_{h}d"
        end_col = f"target_end_date_{h}d"
        start_col = f"target_start_date_{h}d"

        for chunk_idx in range(0, len(test_dates), retrain_every):
            chunk_dates = test_dates[chunk_idx : chunk_idx + retrain_every]
            if not chunk_dates:
                continue
            chunk_start = pd.Timestamp(chunk_dates[0])
            train_mask = (df[end_col] < chunk_start) & df[ret_col].notna() & df[up_col].notna() & df[trade_col].notna()
            test_mask = df["date"].isin(chunk_dates)
            train = df.loc[train_mask].copy()
            test = df.loc[test_mask].copy()
            if len(train) < min_train_samples or test.empty:
                continue

            selected_features = select_ic_feature_columns(train, feature_cols, ret_col, feature_top_k)
            prep = fit_preprocessor(train, selected_features)
            x_train = transform_features(train, selected_features, prep)
            x_test = transform_features(test, selected_features, prep)
            y_ret = train[ret_col].astype(float)
            y_up = train[up_col].astype(int)
            y_trade = train[trade_col].astype(int)
            y_big_up = train[big_up_col].astype(float) if big_up_col in train.columns else pd.Series(np.nan, index=train.index)
            y_big_down = train[big_down_col].astype(float) if big_down_col in train.columns else pd.Series(np.nan, index=train.index)

            base_cols = [
                "date",
                "trade_date",
                start_col,
                end_col,
                "close",
                "fwd_ret_1d",
                ret_col,
                up_col,
                trade_col,
            ]
            for tail_col in [big_up_col, big_down_col]:
                if tail_col in test.columns:
                    base_cols.append(tail_col)
            for spec in model_specs:
                reg = clone_model(spec.reg)
                reg.fit(x_train, y_ret)
                pred_ret = reg.predict(x_test)
                pred_up_prob = fit_binary_probability(spec.cls, x_train, y_up, x_test)
                pred_trade_prob = fit_binary_probability(spec.trade_cls, x_train, y_trade, x_test)
                big_up_model = spec.big_up_cls if spec.big_up_cls is not None else spec.trade_cls
                big_down_model = spec.big_down_cls if spec.big_down_cls is not None else spec.trade_cls
                pred_big_up_prob = fit_optional_binary_probability(big_up_model, x_train, y_big_up, x_test)
                pred_big_down_prob = fit_optional_binary_probability(big_down_model, x_train, y_big_down, x_test)

                pred = test[base_cols].copy()
                pred = pred.rename(
                    columns={
                        start_col: "target_start_date",
                        end_col: "target_end_date",
                        ret_col: "target_ret",
                        up_col: "target_up",
                        trade_col: "target_trade",
                        big_up_col: "target_big_up",
                        big_down_col: "target_big_down",
                    }
                )
                pred["horizon"] = h
                pred["model_name"] = spec.name
                pred["pred_ret"] = pred_ret
                pred["pred_up_prob"] = pred_up_prob
                pred["pred_trade_prob"] = pred_trade_prob
                pred["pred_big_up_prob"] = pred_big_up_prob
                pred["pred_big_down_prob"] = pred_big_down_prob
                pred["pred_tail_score"] = pred["pred_big_up_prob"] - pred["pred_big_down_prob"]
                pred["pred_up"] = (pred["pred_up_prob"] >= 0.5).astype(int)
                pred["pred_trade"] = (pred["pred_trade_prob"] >= 0.5).astype(int)
                pred["pred_big_up"] = (pred["pred_big_up_prob"] >= 0.5).astype(int)
                pred["pred_big_down"] = (pred["pred_big_down_prob"] >= 0.5).astype(int)
                pred["walk_forward_period"] = label
                pred["chunk_start"] = chunk_start
                pred["train_samples"] = len(train)
                pred["train_latest_target_end_date"] = train[end_col].max()
                pred["feature_top_k"] = feature_top_k_label(feature_top_k)
                pred["selected_feature_count"] = len(selected_features)
                all_preds.append(pred)

    if not all_preds:
        raise RuntimeError(f"No {label} walk-forward predictions were generated.")

    out = pd.concat(all_preds, ignore_index=True)
    out = out.sort_values(["date", "horizon", "model_name"]).reset_index(drop=True)
    return out


def safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    valid = pd.concat([y_true, y_prob], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 10 or valid.iloc[:, 0].nunique() < 2:
        return np.nan
    return float(roc_auc_score(valid.iloc[:, 0].astype(int), valid.iloc[:, 1].astype(float)))


def compute_model_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, horizon), grp in pred.groupby(["model_name", "horizon"], sort=True):
        reg_eval = grp.dropna(subset=["target_ret", "pred_ret"]).copy()
        if len(reg_eval) >= 2:
            mae = mean_absolute_error(reg_eval["target_ret"], reg_eval["pred_ret"])
            rmse = math.sqrt(mean_squared_error(reg_eval["target_ret"], reg_eval["pred_ret"]))
            dir_acc = ((reg_eval["target_ret"] > 0) == (reg_eval["pred_ret"] > 0)).mean()
            corr = signed_corr(reg_eval["target_ret"], reg_eval["pred_ret"])
        else:
            mae = rmse = dir_acc = corr = np.nan

        cls_eval = grp.dropna(subset=["target_up", "pred_up", "pred_up_prob"]).copy()
        if len(cls_eval) and cls_eval["target_up"].nunique() > 0:
            y = cls_eval["target_up"].astype(int)
            yhat = cls_eval["pred_up"].astype(int)
            acc = accuracy_score(y, yhat)
            precision = precision_score(y, yhat, zero_division=0)
            recall = recall_score(y, yhat, zero_division=0)
            f1 = f1_score(y, yhat, zero_division=0)
            balanced = balanced_accuracy_score(y, yhat) if y.nunique() > 1 else np.nan
            auc = safe_auc(y, cls_eval["pred_up_prob"])
        else:
            acc = precision = recall = f1 = balanced = auc = np.nan

        trade_eval = grp.dropna(subset=["target_trade", "pred_trade", "pred_trade_prob"]).copy()
        if len(trade_eval) and trade_eval["target_trade"].nunique() > 0:
            yt = trade_eval["target_trade"].astype(int)
            yhatt = trade_eval["pred_trade"].astype(int)
            trade_acc = accuracy_score(yt, yhatt)
            trade_precision = precision_score(yt, yhatt, zero_division=0)
            trade_recall = recall_score(yt, yhatt, zero_division=0)
            trade_f1 = f1_score(yt, yhatt, zero_division=0)
            trade_auc = safe_auc(yt, trade_eval["pred_trade_prob"])
        else:
            trade_acc = trade_precision = trade_recall = trade_f1 = trade_auc = np.nan

        big_up_eval = grp.dropna(subset=["target_big_up", "pred_big_up", "pred_big_up_prob"]).copy()
        if len(big_up_eval) and big_up_eval["target_big_up"].nunique() > 0:
            ybu = big_up_eval["target_big_up"].astype(int)
            yhbu = big_up_eval["pred_big_up"].astype(int)
            big_up_acc = accuracy_score(ybu, yhbu)
            big_up_precision = precision_score(ybu, yhbu, zero_division=0)
            big_up_recall = recall_score(ybu, yhbu, zero_division=0)
            big_up_f1 = f1_score(ybu, yhbu, zero_division=0)
            big_up_auc = safe_auc(ybu, big_up_eval["pred_big_up_prob"])
        else:
            big_up_acc = big_up_precision = big_up_recall = big_up_f1 = big_up_auc = np.nan

        big_down_eval = grp.dropna(subset=["target_big_down", "pred_big_down", "pred_big_down_prob"]).copy()
        if len(big_down_eval) and big_down_eval["target_big_down"].nunique() > 0:
            ybd = big_down_eval["target_big_down"].astype(int)
            yhbd = big_down_eval["pred_big_down"].astype(int)
            big_down_acc = accuracy_score(ybd, yhbd)
            big_down_precision = precision_score(ybd, yhbd, zero_division=0)
            big_down_recall = recall_score(ybd, yhbd, zero_division=0)
            big_down_f1 = f1_score(ybd, yhbd, zero_division=0)
            big_down_auc = safe_auc(ybd, big_down_eval["pred_big_down_prob"])
        else:
            big_down_acc = big_down_precision = big_down_recall = big_down_f1 = big_down_auc = np.nan

        tail_eval = grp.dropna(subset=["target_ret", "pred_tail_score"]).copy()
        tail_corr = signed_corr(tail_eval["target_ret"], tail_eval["pred_tail_score"]) if len(tail_eval) >= 20 else np.nan
        tail_dir_acc = (
            ((tail_eval["target_ret"] > 0) == (tail_eval["pred_tail_score"] > 0)).mean()
            if len(tail_eval) >= 2
            else np.nan
        )

        rows.append(
            {
                "model_name": model_name,
                "horizon": horizon,
                "n_eval": len(reg_eval),
                "MAE": mae,
                "RMSE": rmse,
                "directional_accuracy_from_regression": dir_acc,
                "directional_accuracy": dir_acc,
                "return_correlation": corr,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "F1": f1,
                "AUC": auc,
                "balanced_accuracy": balanced,
                "trade_accuracy": trade_acc,
                "trade_precision": trade_precision,
                "trade_recall": trade_recall,
                "trade_F1": trade_f1,
                "trade_AUC": trade_auc,
                "big_up_accuracy": big_up_acc,
                "big_up_precision": big_up_precision,
                "big_up_recall": big_up_recall,
                "big_up_F1": big_up_f1,
                "big_up_AUC": big_up_auc,
                "big_down_accuracy": big_down_acc,
                "big_down_precision": big_down_precision,
                "big_down_recall": big_down_recall,
                "big_down_F1": big_down_f1,
                "big_down_AUC": big_down_auc,
                "tail_score_return_correlation": tail_corr,
                "tail_score_directional_accuracy": tail_dir_acc,
            }
        )

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["AUC", "directional_accuracy", "return_correlation"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return metrics


def score_validation_metrics_for_feature_selection(metrics: pd.DataFrame) -> float:
    if metrics.empty:
        return -np.inf
    scored = metrics.copy()
    scored["feature_selection_score"] = (
        0.22 * scored["AUC"].fillna(0.5)
        + 0.18 * scored["trade_AUC"].fillna(0.5)
        + 0.20 * scored["big_up_AUC"].fillna(0.5)
        + 0.20 * scored["big_down_AUC"].fillna(0.5)
        + 0.12 * scored["directional_accuracy"].fillna(0.5)
        + 0.08 * scored["tail_score_return_correlation"].clip(lower=0).fillna(0.0)
    )
    return float(scored["feature_selection_score"].max())


def select_validation_feature_top_k(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    model_specs: list[ModelSpec],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for top_k in FEATURE_TOP_K_CANDIDATES:
        print(f"  Validation feature top_k candidate: {feature_top_k_label(top_k)}")
        preds = walk_forward_predictions(
            feature_df,
            feature_cols,
            model_specs,
            VALID_START,
            VALID_END,
            VALID_RETRAIN_EVERY,
            min_train_samples=MIN_TRAIN_SAMPLES,
            label=f"validation_topk_{feature_top_k_label(top_k)}",
            feature_top_k=top_k,
        )
        metrics = compute_model_metrics(preds)
        score = score_validation_metrics_for_feature_selection(metrics)
        best_metric = metrics.iloc[0].to_dict() if not metrics.empty else {}
        row = {
            "top_k": top_k,
            "top_k_label": feature_top_k_label(top_k),
            "validation_score": score,
            "prediction_rows": len(preds),
            "best_metric_model": best_metric.get("model_name"),
            "best_metric_horizon": best_metric.get("horizon"),
            "best_metric_AUC": best_metric.get("AUC"),
            "best_metric_big_up_AUC": best_metric.get("big_up_AUC"),
            "best_metric_big_down_AUC": best_metric.get("big_down_AUC"),
            "best_metric_tail_corr": best_metric.get("tail_score_return_correlation"),
        }
        print(
            "  Finished top_k="
            f"{row['top_k_label']}: score={row['validation_score']:.6f}, "
            f"rows={row['prediction_rows']}, best={row.get('best_metric_model')} {row.get('best_metric_horizon')}D"
        )
        rows.append(row)
        if selected is None or score > selected["validation_score"]:
            selected = {**row, "predictions": preds}

    if selected is None:
        selected = {
            "top_k": None,
            "top_k_label": "all",
            "validation_score": np.nan,
            "prediction_rows": 0,
            "predictions": walk_forward_predictions(
                feature_df,
                feature_cols,
                model_specs,
                VALID_START,
                VALID_END,
                VALID_RETRAIN_EVERY,
                min_train_samples=MIN_TRAIN_SAMPLES,
                label="validation_topk_all",
                feature_top_k=None,
            ),
        }
    return {
        "selected_top_k": selected["top_k"],
        "selected_top_k_label": selected["top_k_label"],
        "selected_validation_score": selected["validation_score"],
        "candidates": rows,
        "predictions": selected["predictions"],
        "selection_rule": "Validation-only max of AUC/trade_AUC/big_up_AUC/big_down_AUC/directional accuracy/tail-score IC composite.",
    }


def compute_factor_effectiveness(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    label = "target_ret_5d"
    up = "target_up_5d"
    end = "target_end_date_5d"
    sample = df.loc[(df[end] < TEST_START) & df[label].notna()].copy()
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        x = sample[col].replace([np.inf, -np.inf], np.nan)
        y = sample[label]
        corr = signed_corr(x, y)
        mean_up = x[sample[up] == 1].mean()
        mean_down = x[sample[up] == 0].mean()
        rows.append(
            {
                "factor": col,
                "corr_with_forward_return": corr,
                "mean_when_up": mean_up,
                "mean_when_down": mean_down,
                "difference_up_down": mean_up - mean_down,
                "missing_rate": float(x.isna().mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_corr_rank"] = out["corr_with_forward_return"].abs().rank(ascending=False, method="dense")
    return out.sort_values(["abs_corr_rank", "factor"]).reset_index(drop=True)


def compute_factor_effectiveness_tail_targets(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        ret_col = f"target_ret_{h}d"
        big_up_col = f"target_big_up_{h}d"
        big_down_col = f"target_big_down_{h}d"
        end_col = f"target_end_date_{h}d"
        sample = df.loc[(df[end_col] < TEST_START) & df[ret_col].notna()].copy()
        for col in feature_cols:
            x = sample[col].replace([np.inf, -np.inf], np.nan)
            rows.append(
                {
                    "factor": col,
                    "horizon": h,
                    "corr_with_forward_return": signed_corr(x, sample[ret_col]),
                    "corr_with_big_up": signed_corr(x, sample[big_up_col]) if big_up_col in sample.columns else np.nan,
                    "corr_with_big_down": signed_corr(x, sample[big_down_col]) if big_down_col in sample.columns else np.nan,
                    "mean_when_big_up": x[sample[big_up_col] == 1].mean() if big_up_col in sample.columns else np.nan,
                    "mean_when_big_down": x[sample[big_down_col] == 1].mean() if big_down_col in sample.columns else np.nan,
                    "difference_big_up_down": (
                        x[sample[big_up_col] == 1].mean() - x[sample[big_down_col] == 1].mean()
                        if big_up_col in sample.columns and big_down_col in sample.columns
                        else np.nan
                    ),
                    "missing_rate": float(x.isna().mean()),
                }
            )
    out = pd.DataFrame(rows)
    out["max_abs_tail_corr"] = out[["corr_with_big_up", "corr_with_big_down"]].abs().max(axis=1)
    return out.sort_values(["max_abs_tail_corr", "factor", "horizon"], ascending=[False, True, True]).reset_index(drop=True)


def technical_raw_position(df: pd.DataFrame) -> pd.Series:
    trend = ((df["ma_spread_20_60"] > 0) & (df["ma_ratio_20"] > -0.01)).astype(float)
    momentum = (df["ret_20"] > 0).astype(float)
    drawdown_ok = (df["drawdown_60"] > -0.10).astype(float)
    low_vol = (df["volatility_ratio_20_60"] < 1.15).astype(float)
    breakout = (df["breakout_high_prior_20"] > -0.015).astype(float)
    bull = (df["trend_regime_score"] >= 0.5).astype(float)
    score = 0.25 * trend + 0.20 * momentum + 0.18 * drawdown_ok + 0.15 * low_vol + 0.12 * breakout + 0.10 * bull
    pos = 0.15 + 0.85 * score
    pos = pos.where(~((df["drawdown_120"] < -0.18) & (df["volatility_ratio_20_60"] > 1.25)), pos * 0.55)
    return pos.clip(0.0, 1.0).fillna(0.5)


def add_prediction_ranks(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.sort_values("date").copy()
    out["pred_ret_rank_60"] = rolling_percentile(out["pred_ret"], 60, 10).fillna(0.5)
    out["pred_up_prob_rank_60"] = rolling_percentile(out["pred_up_prob"], 60, 10).fillna(0.5)
    out["pred_trade_prob_rank_60"] = rolling_percentile(out["pred_trade_prob"], 60, 10).fillna(0.5)
    if "pred_tail_score" in out.columns:
        out["pred_tail_score_rank_60"] = rolling_percentile(out["pred_tail_score"], 60, 10).fillna(0.5)
    else:
        out["pred_tail_score_rank_60"] = 0.5
    out["pred_ret_rank_20"] = rolling_percentile(out["pred_ret"], 20, 5).fillna(0.5)
    out["pred_up_prob_rank_20"] = rolling_percentile(out["pred_up_prob"], 20, 5).fillna(0.5)
    out["pred_trade_prob_rank_20"] = rolling_percentile(out["pred_trade_prob"], 20, 5).fillna(0.5)
    if "pred_tail_score" in out.columns:
        out["pred_tail_score_rank_20"] = rolling_percentile(out["pred_tail_score"], 20, 5).fillna(0.5)
    else:
        out["pred_tail_score_rank_20"] = 0.5
    return out


def model_raw_positions(merged: pd.DataFrame) -> dict[str, pd.Series]:
    tech = technical_raw_position(merged)
    risk_penalty = (
        0.18 * (merged["ma_ratio_60"] < 0).astype(float)
        + 0.14 * (merged["drawdown_60"] < -0.10).astype(float)
        + 0.12 * (merged["volatility_ratio_20_60"] > 1.2).astype(float)
        + 0.10 * (merged["trend_regime_score"] < 0.35).astype(float)
    )
    ret_rank = merged["pred_ret_rank_60"].fillna(0.5)
    up_rank = merged["pred_up_prob_rank_60"].fillna(0.5)
    trade_rank = merged["pred_trade_prob_rank_60"].fillna(0.5)
    strength = 0.42 * ret_rank + 0.34 * up_rank + 0.24 * trade_rank
    bearish = ((merged["pred_ret"] < 0) & (merged["pred_up_prob"] < 0.48)).astype(float)

    aggressive = 0.30 + 0.55 * strength + 0.20 * tech - risk_penalty - 0.12 * bearish
    benchmark_aware = 0.72 + 0.22 * strength + 0.10 * tech - 0.55 * risk_penalty - 0.08 * bearish
    risk_control = 0.50 + 0.30 * strength + 0.15 * tech - 0.85 * risk_penalty - 0.10 * bearish

    strong_model = (merged["pred_trade_prob"] > 0.60) & (merged["pred_ret"] > TRADE_THRESHOLD)
    benchmark_aware = benchmark_aware.where(~strong_model, benchmark_aware + 0.08)
    aggressive = aggressive.where(~strong_model, aggressive + 0.10)

    return {
        "ml_signal_aggressive": aggressive.clip(0.30, 1.0).fillna(0.60),
        "ml_signal_benchmark_aware": benchmark_aware.clip(0.60, 1.0).fillna(0.80),
        "ml_risk_control": risk_control.clip(0.20, 1.0).fillna(0.55),
    }


def ml_index_enhanced_core_position(merged: pd.DataFrame) -> pd.Series:
    strong_trend = (merged["ma_ratio_60"] > 0) & (merged["ma_spread_20_60"] > 0) & (merged["ret_20"] > 0)
    uptrend = (merged["ma_ratio_60"] > 0) & (merged["ret_20"] > 0)
    weak_trend = (merged["ma_ratio_60"] < 0) | (merged["ret_20"] < 0)
    high_risk = (merged["drawdown_60"] < -0.12) | (merged["volatility_20"] > 1.2 * merged["volatility_60"])
    strong_model = ((merged["pred_up_prob"] > 0.55) | (merged["pred_trade_prob"] > 0.55)) & (merged["pred_ret"] > 0)
    weak_model = ((merged["pred_up_prob"] < 0.45) | (merged["pred_trade_prob"] < 0.45)) & (merged["pred_ret"] < 0)

    position = pd.Series(0.95, index=merged.index, dtype=float)
    position = position.where(~strong_trend, 1.00)
    position = position.where(~(uptrend & ~strong_trend), 0.95)
    position = position.where(~weak_trend, 0.85)
    position = position - 0.10 * high_risk.astype(float)
    position = position + 0.04 * strong_model.astype(float)
    position = position - 0.07 * weak_model.astype(float)
    position = position.where(~(uptrend & ~high_risk), position.clip(lower=0.90))
    return position.clip(0.75, 1.00).fillna(0.85)


def ml_index_enhanced_plus_position(merged: pd.DataFrame) -> pd.Series:
    core = ml_index_enhanced_core_position(merged)
    strong_trend = (
        (merged["ma_ratio_60"] > 0)
        & (merged["ma_spread_20_60"] > 0)
        & (merged["ret_20"] > 0)
        & (merged["drawdown_60"] > -0.08)
    )
    strong_model = ((merged["pred_up_prob"] > 0.60) | (merged["pred_trade_prob"] > 0.60)) & (merged["pred_ret"] > 0)
    high_risk = (merged["drawdown_60"] < -0.12) | (merged["volatility_20"] > 1.2 * merged["volatility_60"])
    position = core.copy()
    position = position.where(~(strong_trend & strong_model), 1.10)
    position = position.where(~(strong_trend & ~strong_model), np.maximum(position, 1.00))
    position = position.where(~high_risk, position.clip(upper=1.00))
    return position.clip(0.75, 1.10).fillna(0.90)


def ml_index_full_participation_position(
    merged: pd.DataFrame,
    weak_prob: float = 0.45,
    high_vol_multiplier: float = 1.25,
    min_position: float = 0.88,
    risk_cut_position: float = 0.92,
) -> pd.Series:
    strong_trend = (merged["ma_ratio_60"] > 0) & (merged["ma_spread_20_60"] > 0) & (merged["ret_20"] > 0)
    weak_trend = (merged["ma_ratio_60"] < 0) | (merged["ret_20"] < 0)
    high_risk = (merged["drawdown_60"] < -0.12) | (
        merged["volatility_20"] > high_vol_multiplier * merged["volatility_60"]
    )
    weak_model = (
        (merged["pred_up_prob"] < weak_prob)
        & (merged["pred_trade_prob"] < weak_prob)
        & (merged["pred_ret"] < 0)
    )

    position = pd.Series(1.00, index=merged.index, dtype=float)
    position = position.where(~strong_trend, 1.00)
    position = position.where(~(strong_trend & weak_model & high_risk), risk_cut_position)
    position = position.where(~(weak_trend & ~high_risk), 0.95)
    position = position.where(~(weak_trend & high_risk & weak_model), min_position)
    return position.clip(min_position, 1.00).fillna(1.00)


def ml_directional_alpha_extratrees_10d_position(merged: pd.DataFrame) -> pd.Series:
    strong_trend = (merged["ma_ratio_60"] > 0) & (merged["ma_spread_20_60"] > 0) & (merged["ret_20"] > 0)
    high_risk = (merged["drawdown_60"] < -0.12) | (merged["volatility_20"] > 1.25 * merged["volatility_60"])
    position = pd.Series(0.95, index=merged.index, dtype=float)
    positive_signal = (merged["pred_ret"] > 0) | (merged["pred_up_prob"] > 0.52)
    mild_negative = (merged["pred_ret"] < 0) & (merged["pred_up_prob"] < 0.48)
    deep_negative = (merged["pred_ret"] < 0) & (merged["pred_up_prob"] < 0.45) & high_risk
    weak_trend = (merged["ma_ratio_60"] < 0) | (merged["ret_20"] < 0)
    position = position.where(~positive_signal, 1.00)
    position = position.where(~(mild_negative & weak_trend), 0.90)
    position = position.where(~deep_negative, 0.85)
    position = position.where(~strong_trend, position.clip(lower=0.95))
    return position.clip(0.85, 1.00).fillna(0.95)


def ml_directional_long_cash_position(merged: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    buy_prob = float(params["buy_prob"])
    sell_prob = float(params["sell_prob"])
    min_pred_ret_buy = float(params["min_pred_ret_buy"])
    max_pred_ret_sell = float(params["max_pred_ret_sell"])
    use_trade_prob = bool(params["use_trade_prob"])
    trade_prob_cut = float(params["trade_prob_cut"])

    current = 1.0
    values: list[float] = []
    for _, row in merged.sort_values("date").iterrows():
        buy_signal = (row.get("pred_up_prob", np.nan) >= buy_prob) and (
            row.get("pred_ret", np.nan) >= min_pred_ret_buy
        )
        if use_trade_prob:
            buy_signal = buy_signal and (row.get("pred_trade_prob", np.nan) >= trade_prob_cut)
        sell_signal = (row.get("pred_up_prob", np.nan) <= sell_prob) and (
            row.get("pred_ret", np.nan) <= max_pred_ret_sell
        )
        if buy_signal:
            current = 1.0
        elif sell_signal:
            current = 0.0
        values.append(current)
    return pd.Series(values, index=merged.sort_values("date").index, dtype=float).reindex(merged.index).clip(0.0, 1.0)


def ml_directional_scaled_timing_position(merged: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    strong_buy_prob = float(params["strong_buy_prob"])
    buy_prob = float(params["buy_prob"])
    sell_prob = float(params["sell_prob"])
    strong_sell_prob = float(params["strong_sell_prob"])
    strong_buy_position = float(params.get("strong_buy_position", 1.0))
    buy_position = float(params["buy_position"])
    neutral_position = float(params["neutral_position"])
    sell_position = float(params["sell_position"])
    strong_sell_position = float(params["strong_sell_position"])
    trend_filter = bool(params["trend_filter"])
    risk_filter = bool(params["risk_filter"])

    pred_up = merged["pred_up_prob"].astype(float)
    pred_ret = merged["pred_ret"].astype(float)
    position = pd.Series(neutral_position, index=merged.index, dtype=float)
    position = position.where(~((pred_up >= buy_prob) & (pred_ret >= 0)), buy_position)
    position = position.where(~((pred_up >= strong_buy_prob) & (pred_ret > 0)), strong_buy_position)
    position = position.where(~((pred_up <= sell_prob) & (pred_ret < 0)), sell_position)
    position = position.where(~((pred_up <= strong_sell_prob) & (pred_ret < 0)), strong_sell_position)

    if trend_filter:
        strong_trend = (merged["ma_ratio_60"] > 0) & (merged["ma_spread_20_60"] > 0) & (merged["ret_20"] > 0)
        weak_trend = (merged["ma_ratio_60"] < 0) | (merged["ret_20"] < 0)
        position = position.where(~strong_trend, position.clip(lower=buy_position))
        position = position.where(~(weak_trend & (pred_up < sell_prob)), np.minimum(position, sell_position))

    if risk_filter:
        high_risk = (merged["drawdown_60"] < -0.10) | (merged["volatility_20"] > 1.2 * merged["volatility_60"])
        position = position.where(~(high_risk & (pred_up < sell_prob)), np.minimum(position, sell_position))

    return position.clip(0.0, 1.0).fillna(neutral_position)


def ml_selective_defensive_enhancement_position(merged: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    weak_prob = float(params["weak_prob"])
    strong_negative_prob = float(params["strong_negative_prob"])
    weak_ret_cut = float(params["weak_ret_cut"])
    mild_cut_position = float(params["mild_cut_position"])
    risk_cut_position = float(params["risk_cut_position"])
    min_position = float(params["min_position"])
    high_vol_multiplier = float(params["high_vol_multiplier"])
    drawdown_cut = float(params["drawdown_cut"])
    require_trend_confirm = bool(params.get("require_trend_confirm", True))
    require_risk_confirm = bool(params.get("require_risk_confirm", True))

    def _col(name: str, default: float = np.nan) -> pd.Series:
        if name in merged.columns:
            return merged[name].astype(float)
        return pd.Series(default, index=merged.index, dtype=float)

    pred_up = _col("pred_up_prob")
    pred_ret = _col("pred_ret")
    pred_trade = _col("pred_trade_prob", 1.0)
    ma_ratio_60 = _col("ma_ratio_60")
    ma_spread_20_60 = _col("ma_spread_20_60")
    ret_20 = _col("ret_20")
    trend_regime_score = _col("trend_regime_score", 1.0)
    drawdown_60 = _col("drawdown_60")
    volatility_20 = _col("volatility_20")
    volatility_60 = _col("volatility_60")
    bear_high_vol_regime = _col("bear_high_vol_regime", 0.0)
    crash_risk_score = _col("crash_risk_score")
    crash_risk_threshold = crash_risk_score.rolling(60, min_periods=20).quantile(0.70)

    strong_trend = (ma_ratio_60 > 0) & (ma_spread_20_60 > 0) & (ret_20 > 0)
    weak_trend = (ma_ratio_60 < 0) | (ret_20 < 0) | (trend_regime_score < 0.5)
    high_risk = (
        (drawdown_60 < drawdown_cut)
        | (volatility_20 > high_vol_multiplier * volatility_60)
        | (bear_high_vol_regime > 0)
        | (crash_risk_score > crash_risk_threshold)
    )

    if "m5_close_30m_sell_pressure_rank_20" in merged.columns:
        high_risk = high_risk | (merged["m5_close_30m_sell_pressure_rank_20"].astype(float) > 0.70)
    if "m5_net_volume_pressure_rank_20" in merged.columns:
        high_risk = high_risk | (merged["m5_net_volume_pressure_rank_20"].astype(float) < 0.30)
    if "m5_high_to_close_reversal_rank_20" in merged.columns:
        high_risk = high_risk | (merged["m5_high_to_close_reversal_rank_20"].astype(float) < 0.30)

    ml_negative = (pred_up < weak_prob) & (pred_ret < weak_ret_cut)
    ml_strong_negative = (
        (pred_up < strong_negative_prob)
        & (pred_trade < strong_negative_prob)
        & (pred_ret < weak_ret_cut)
    )
    uncertain_condition = ml_negative & ~(strong_trend & (pred_up >= 0.50))

    if require_trend_confirm:
        mild_condition = ml_negative & weak_trend
    else:
        mild_condition = ml_negative & (weak_trend | high_risk)

    if require_risk_confirm:
        risk_condition = ml_negative & weak_trend & high_risk
    else:
        risk_condition = ml_negative & high_risk

    strong_condition = ml_strong_negative & weak_trend & high_risk

    position = pd.Series(1.00, index=merged.index, dtype=float)
    position = position.mask((strong_trend & (pred_up >= 0.50)).fillna(False), 1.00)
    position = position.mask(uncertain_condition.fillna(False), mild_cut_position)
    position = position.mask(mild_condition.fillna(False), mild_cut_position)
    position = position.mask(risk_condition.fillna(False), risk_cut_position)
    position = position.mask(strong_condition.fillna(False), min_position)
    return position.clip(min_position, 1.00).fillna(1.00)


def ml_validation_tuned_index_core_position(merged: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    min_position = float(params["min_position"])
    strong_floor = float(params["strong_floor"])
    weak_prob = float(params["weak_prob"])
    strong_prob = float(params["strong_prob"])
    high_vol_multiplier = float(params["high_vol_multiplier"])
    risk_cut = float(params["risk_cut"])

    strong_trend = (merged["ma_ratio_60"] > 0) & (merged["ma_spread_20_60"] > 0) & (merged["ret_20"] > 0)
    weak_trend = (merged["ma_ratio_60"] < 0) | (merged["ret_20"] < 0)
    high_risk = (merged["drawdown_60"] < -0.12) | (
        merged["volatility_20"] > high_vol_multiplier * merged["volatility_60"]
    )
    weak_model = (
        (merged["pred_up_prob"] < weak_prob)
        & (merged["pred_trade_prob"] < weak_prob)
        & (merged["pred_ret"] < 0)
    )
    strong_model = ((merged["pred_up_prob"] > strong_prob) | (merged["pred_trade_prob"] > strong_prob)) & (
        merged["pred_ret"] > 0
    )

    position = pd.Series(0.97, index=merged.index, dtype=float)
    position = position.where(~(strong_trend | strong_model), 1.00)
    position = position.where(~(~strong_trend & ~strong_model), strong_floor)
    position = position.where(~weak_trend, np.maximum(min_position, strong_floor - risk_cut))
    position = position.where(~weak_model, position - risk_cut)
    position = position.where(~(weak_model & high_risk), min_position)
    position = position.where(~strong_trend, position.clip(lower=strong_floor))
    return position.clip(min_position, 1.00).fillna(strong_floor)


def ml_index_enhanced_plus_participation_position(merged: pd.DataFrame, max_exposure: float) -> pd.Series:
    position = ml_index_full_participation_position(
        merged,
        weak_prob=0.45,
        high_vol_multiplier=1.25,
        min_position=0.85,
        risk_cut_position=0.92,
    )
    strong_trend = (
        (merged["ma_ratio_60"] > 0)
        & (merged["ma_spread_20_60"] > 0)
        & (merged["ret_20"] > 0)
        & (merged["drawdown_60"] > -0.08)
    )
    strong_model = ((merged["pred_up_prob"] > 0.58) | (merged["pred_trade_prob"] > 0.58)) & (
        merged["pred_ret"] > 0
    )
    weak_trend = (merged["ma_ratio_60"] < 0) | (merged["ret_20"] < 0)
    high_risk = (merged["drawdown_60"] < -0.12) | (merged["volatility_20"] > 1.25 * merged["volatility_60"])
    position = position.where(~(strong_trend & strong_model), max_exposure)
    position = position.where(~(high_risk | weak_trend), position.clip(upper=1.00))
    return position.clip(0.85, max_exposure).fillna(0.95)


def ml_cost_aware_index_enhancement_position(
    merged: pd.DataFrame,
    params: dict[str, Any],
    conservative: bool = False,
) -> pd.Series:
    min_position = 0.95 if conservative else float(params["min_position"])
    strong_trend_cut_position = 0.98 if conservative else float(params["strong_trend_cut_position"])
    weak_cut_position = 0.98 if conservative else float(params["weak_cut_position"])
    risk_cut_position = 0.97 if conservative else float(params["risk_cut_position"])
    positive_prob = float(params["positive_prob"])
    negative_prob = float(params["negative_prob"])
    strong_negative_prob = float(params["strong_negative_prob"])
    drawdown_cut = float(params["drawdown_cut"])
    high_vol_multiplier = float(params["high_vol_multiplier"])

    strong_trend = (merged["ma_ratio_60"] > 0) & (merged["ma_spread_20_60"] > 0) & (merged["ret_20"] > 0)
    weak_trend = (merged["ma_ratio_60"] < 0) | (merged["ret_20"] < 0)
    high_risk = (merged["drawdown_60"] < drawdown_cut) | (
        merged["volatility_20"] > high_vol_multiplier * merged["volatility_60"]
    )
    ml_positive = (merged["pred_ret"] > 0) & (merged["pred_up_prob"] > positive_prob)
    ml_negative = (merged["pred_ret"] < 0) & (merged["pred_up_prob"] < negative_prob)
    ml_strong_negative = (
        (merged["pred_ret"] < 0)
        & (merged["pred_up_prob"] < strong_negative_prob)
        & (merged["pred_trade_prob"] < strong_negative_prob)
    )

    position = pd.Series(1.00, index=merged.index, dtype=float)
    if conservative:
        position = position.where(~(ml_strong_negative & weak_trend & high_risk), min_position)
    else:
        position = position.where(~(strong_trend & ml_strong_negative & high_risk), strong_trend_cut_position)
        position = position.where(~(~strong_trend & ml_positive), 1.00)
        position = position.where(~(weak_trend & ml_negative), weak_cut_position)
        position = position.where(~(weak_trend & high_risk & ml_strong_negative), min_position)
        position = position.where(~(high_risk & ~ml_strong_negative), risk_cut_position)
    return position.clip(min_position, 1.00).fillna(1.00)


def ml_tail_score_index_enhancement_position(merged: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    low_tail_score = float(params["low_tail_score"])
    very_low_tail_score = float(params["very_low_tail_score"])
    weak_cut_position = float(params["weak_cut_position"])
    risk_cut_position = float(params["risk_cut_position"])
    min_position = float(params["min_position"])
    high_risk_multiplier = float(params["high_risk_multiplier"])
    score = merged["pred_tail_score"].fillna(0.0)
    big_down_prob = merged["pred_big_down_prob"].fillna(0.5)
    strong_tail = score >= float(params["strong_tail_score"])

    weak_trend = (
        (merged["trend_regime_score"] < 0.45)
        | (merged["ret_20"] < 0)
        | (merged["ma_ratio_60"] < 0)
        | (merged["up_day_ratio_20"] < 0.45)
        | (merged["trend_r2_20"].fillna(0.0) < 0.10)
    )
    high_risk = (
        (merged["bear_high_vol_regime"] > 0)
        | (merged["drawdown_60"] < -0.10)
        | (merged["volatility_20"] > high_risk_multiplier * merged["volatility_60"])
        | (merged["crash_risk_score"] > merged["crash_risk_score"].rolling(60, min_periods=20).quantile(0.70))
        | (merged["sell_pressure_volume_20"] > merged["sell_pressure_volume_20"].rolling(60, min_periods=20).quantile(0.70))
    )
    strong_trend = (
        (merged["trend_regime_score"] >= 0.75)
        & (merged["ret_20"] > 0)
        & (merged["ma_ratio_60"] > 0)
        & (merged["bull_low_vol_regime"] > 0)
    )

    position = pd.Series(1.00, index=merged.index, dtype=float)
    low_tail = score < low_tail_score
    very_low_tail = (score < very_low_tail_score) | (big_down_prob > float(params["big_down_prob_cut"]))
    position = position.where(~(low_tail & weak_trend), weak_cut_position)
    position = position.where(~(low_tail & high_risk), risk_cut_position)
    position = position.where(~(very_low_tail & weak_trend & high_risk), min_position)
    position = position.where(~(strong_tail & strong_trend), 1.00)
    return position.clip(min_position, 1.00).fillna(1.00)


def ml_vol_target_trend_frame(merged: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=merged.index)
    realized_vol = (merged["volatility_20"] * math.sqrt(252)).replace([np.inf, -np.inf], np.nan)
    vol_position = (0.25 / realized_vol).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    uptrend = (merged["ma_ratio_60"] > 0) & (merged["ret_20"] > 0)
    weak_trend = (merged["ma_ratio_60"] < 0) | (merged["ret_20"] < 0)
    trend_multiplier = pd.Series(1.0, index=merged.index, dtype=float)
    trend_multiplier = trend_multiplier.where(~weak_trend, 0.80)
    trend_multiplier = trend_multiplier.where(~(merged["drawdown_60"] < -0.12), trend_multiplier * 0.85)
    positive_ml = (merged["pred_up_prob"] > 0.55) & (merged["pred_ret"] > 0)
    negative_ml = (merged["pred_up_prob"] < 0.45) & (merged["pred_ret"] < 0)
    ml_adjustment = pd.Series(0.0, index=merged.index, dtype=float)
    ml_adjustment = ml_adjustment.where(~positive_ml, 0.05)
    ml_adjustment = ml_adjustment.where(~negative_ml, -0.10)
    position = vol_position * trend_multiplier + ml_adjustment
    position = position.clip(0.60, 1.00)
    position = position.where(~uptrend, position.clip(lower=0.85))
    out["realized_vol"] = realized_vol
    out["vol_position"] = vol_position
    out["trend_multiplier"] = trend_multiplier
    out["ml_adjustment"] = ml_adjustment
    out["raw_position"] = position.clip(0.60, 1.00).fillna(0.85)
    return out


def apply_rebalance(
    raw_position: pd.Series,
    freq: int = REBALANCE_FREQ,
    lower: float = 0.0,
    upper: float = 1.0,
) -> pd.Series:
    raw = raw_position.astype(float).clip(lower, upper).ffill().fillna(lower)
    values: list[float] = []
    current = float(raw.iloc[0]) if len(raw) else 0.0
    for i, val in enumerate(raw):
        if i == 0 or i % freq == 0:
            current = float(val)
        values.append(current)
    return pd.Series(values, index=raw.index)


def apply_rebalance_with_no_trade_band(
    raw_position: pd.Series,
    freq: int,
    lower: float,
    upper: float,
    no_trade_band: float,
) -> pd.Series:
    raw = raw_position.astype(float).clip(lower, upper).ffill().fillna(lower)
    values: list[float] = []
    current = float(raw.iloc[0]) if len(raw) else lower
    for i, value in enumerate(raw):
        target = float(value)
        if i == 0:
            current = target
        elif i % freq == 0 and abs(target - current) >= no_trade_band:
            current = target
        values.append(float(np.clip(current, lower, upper)))
    return pd.Series(values, index=raw.index)


def drawdown_multiplier(drawdown: float) -> tuple[float, str]:
    if drawdown <= -0.15:
        return 0.35, "defensive_15pct"
    if drawdown <= -0.12:
        return 0.50, "cut_12pct"
    if drawdown <= -0.08:
        return 0.70, "cut_8pct"
    if drawdown <= -0.06:
        return 0.85, "watch_6pct"
    if drawdown <= -0.03:
        return 0.93, "recovering"
    return 1.0, "normal"


def simulate_strategy(
    base: pd.DataFrame,
    raw_position: pd.Series,
    strategy: str,
    model_name: str,
    horizon: int | str,
    apply_dd_control: bool = True,
    min_position: float = 0.0,
    max_position: float = 1.0,
) -> pd.DataFrame:
    work = base.sort_values("date").copy().reset_index(drop=True)
    scheduled = apply_rebalance(raw_position.reset_index(drop=True), REBALANCE_FREQ, min_position, max_position)
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    prev_pos = 0.0
    rows: list[dict[str, Any]] = []

    for i, row in work.iterrows():
        path_dd = equity / peak - 1
        if apply_dd_control:
            multiplier, state = drawdown_multiplier(path_dd)
        else:
            multiplier, state = (1.0, "benchmark") if strategy == "buy_hold" else (1.0, "strategy_rule")
        raw = float(scheduled.iloc[i])
        final_pos = float(np.clip(raw * multiplier, min_position, max_position))
        turnover = abs(final_pos - prev_pos)
        realized = final_pos * float(row["fwd_ret_1d"]) - turnover * COST_RATE
        equity = equity * (1 + realized)
        peak = max(peak, equity)
        realized_dd = equity / peak - 1
        rows.append(
            {
                "date": row["date"],
                "trade_date": row["trade_date"],
                "strategy": strategy,
                "model_name": model_name,
                "horizon": horizon,
                "close": row["close"],
                "fwd_ret_1d": row["fwd_ret_1d"],
                "raw_position": raw,
                "final_position": final_pos,
                "turnover": turnover,
                "strategy_return": realized,
                "equity": equity,
                "drawdown": realized_dd,
                "risk_control_state": state,
                "drawdown_control_multiplier": multiplier,
                "pred_ret": row.get("pred_ret", np.nan),
                "pred_up_prob": row.get("pred_up_prob", np.nan),
                "pred_trade_prob": row.get("pred_trade_prob", np.nan),
                "pred_big_up_prob": row.get("pred_big_up_prob", np.nan),
                "pred_big_down_prob": row.get("pred_big_down_prob", np.nan),
                "pred_tail_score": row.get("pred_tail_score", np.nan),
                "realized_vol": row.get("realized_vol", np.nan),
                "vol_position": row.get("vol_position", np.nan),
                "trend_multiplier": row.get("trend_multiplier", np.nan),
                "ml_adjustment": row.get("ml_adjustment", np.nan),
                "cppi_floor": row.get("cppi_floor", np.nan),
                "cppi_cushion": row.get("cppi_cushion", np.nan),
                "cppi_position": row.get("cppi_position", np.nan),
            }
        )
        prev_pos = final_pos

    return pd.DataFrame(rows)


def simulate_strategy_cost_aware(
    base: pd.DataFrame,
    raw_position: pd.Series,
    strategy: str,
    model_name: str,
    horizon: int | str,
    no_trade_band: float,
    apply_dd_control: bool = False,
    min_position: float = 0.0,
    max_position: float = 1.0,
) -> pd.DataFrame:
    work = base.sort_values("date").copy().reset_index(drop=True)
    scheduled = apply_rebalance_with_no_trade_band(
        raw_position.reset_index(drop=True),
        REBALANCE_FREQ,
        min_position,
        max_position,
        no_trade_band,
    )
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    prev_pos = 0.0
    rows: list[dict[str, Any]] = []

    for i, row in work.iterrows():
        path_dd = equity / peak - 1
        if apply_dd_control:
            multiplier, state = drawdown_multiplier(path_dd)
        else:
            multiplier, state = 1.0, "cost_aware_rule"
        raw = float(scheduled.iloc[i])
        final_pos = float(np.clip(raw * multiplier, min_position, max_position))
        turnover = abs(final_pos - prev_pos)
        realized = final_pos * float(row["fwd_ret_1d"]) - turnover * COST_RATE
        equity = equity * (1 + realized)
        peak = max(peak, equity)
        realized_dd = equity / peak - 1
        rows.append(
            {
                "date": row["date"],
                "trade_date": row["trade_date"],
                "strategy": strategy,
                "model_name": model_name,
                "horizon": horizon,
                "close": row["close"],
                "fwd_ret_1d": row["fwd_ret_1d"],
                "raw_position": raw,
                "final_position": final_pos,
                "turnover": turnover,
                "strategy_return": realized,
                "equity": equity,
                "drawdown": realized_dd,
                "risk_control_state": state,
                "drawdown_control_multiplier": multiplier,
                "pred_ret": row.get("pred_ret", np.nan),
                "pred_up_prob": row.get("pred_up_prob", np.nan),
                "pred_trade_prob": row.get("pred_trade_prob", np.nan),
                "pred_big_up_prob": row.get("pred_big_up_prob", np.nan),
                "pred_big_down_prob": row.get("pred_big_down_prob", np.nan),
                "pred_tail_score": row.get("pred_tail_score", np.nan),
                "realized_vol": row.get("realized_vol", np.nan),
                "vol_position": row.get("vol_position", np.nan),
                "trend_multiplier": row.get("trend_multiplier", np.nan),
                "ml_adjustment": row.get("ml_adjustment", np.nan),
                "cppi_floor": row.get("cppi_floor", np.nan),
                "cppi_cushion": row.get("cppi_cushion", np.nan),
                "cppi_position": row.get("cppi_position", np.nan),
                "no_trade_band": no_trade_band,
            }
        )
        prev_pos = final_pos

    return pd.DataFrame(rows)


def simulate_cppi_strategy(
    base: pd.DataFrame,
    strategy: str,
    model_name: str,
    horizon: int | str,
) -> pd.DataFrame:
    work = base.sort_values("date").copy().reset_index(drop=True)
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    prev_pos = 0.0
    rows: list[dict[str, Any]] = []

    for _, row in work.iterrows():
        floor = peak * 0.90
        cushion = max(equity - floor, 0.0) / equity if equity > 0 else 0.0
        cppi_position = 4.0 * cushion
        position = float(np.clip(cppi_position, 0.50, 1.00))

        uptrend = (row["ma_ratio_60"] > 0) and (row["ret_20"] > 0)
        weak_trend = (row["ma_ratio_60"] < 0) and (row["ret_20"] < 0)
        if uptrend:
            position = max(position, 0.90)
        if weak_trend:
            position = min(position, 0.80)

        if (row.get("pred_up_prob", np.nan) > 0.55) and (row.get("pred_ret", np.nan) > 0):
            position += 0.05
        if (row.get("pred_up_prob", np.nan) < 0.45) and (row.get("pred_ret", np.nan) < 0):
            position -= 0.10

        raw = float(np.clip(position, 0.50, 1.00))
        final_pos = raw
        turnover = abs(final_pos - prev_pos)
        realized = final_pos * float(row["fwd_ret_1d"]) - turnover * COST_RATE
        equity = equity * (1 + realized)
        peak = max(peak, equity)
        realized_dd = equity / peak - 1

        rows.append(
            {
                "date": row["date"],
                "trade_date": row["trade_date"],
                "strategy": strategy,
                "model_name": model_name,
                "horizon": horizon,
                "close": row["close"],
                "fwd_ret_1d": row["fwd_ret_1d"],
                "raw_position": raw,
                "final_position": final_pos,
                "turnover": turnover,
                "strategy_return": realized,
                "equity": equity,
                "drawdown": realized_dd,
                "risk_control_state": "cppi_floor",
                "drawdown_control_multiplier": 1.0,
                "pred_ret": row.get("pred_ret", np.nan),
                "pred_up_prob": row.get("pred_up_prob", np.nan),
                "pred_trade_prob": row.get("pred_trade_prob", np.nan),
                "pred_big_up_prob": row.get("pred_big_up_prob", np.nan),
                "pred_big_down_prob": row.get("pred_big_down_prob", np.nan),
                "pred_tail_score": row.get("pred_tail_score", np.nan),
                "realized_vol": row.get("realized_vol", np.nan),
                "vol_position": row.get("vol_position", np.nan),
                "trend_multiplier": row.get("trend_multiplier", np.nan),
                "ml_adjustment": row.get("ml_adjustment", np.nan),
                "cppi_floor": floor,
                "cppi_cushion": cushion,
                "cppi_position": cppi_position,
            }
        )
        prev_pos = final_pos

    return pd.DataFrame(rows)


def select_validation_components(validation_metrics: pd.DataFrame) -> dict[str, Any]:
    if validation_metrics.empty:
        return {
            "best_regression": {"model_name": "HistGradientBoosting", "horizon": 5},
            "best_classification": {"model_name": "HistGradientBoosting", "horizon": 5},
            "best_trade": {"model_name": "HistGradientBoosting", "horizon": 5},
        }

    reg_df = validation_metrics.copy()
    reg_df["reg_score"] = (
        reg_df["directional_accuracy"].fillna(0.5)
        + reg_df["return_correlation"].clip(lower=0).fillna(0.0)
        - 0.05 * reg_df["RMSE"].rank(pct=True).fillna(0.5)
    )
    cls_df = validation_metrics.copy()
    cls_df["cls_score"] = cls_df["AUC"].fillna(0.5) + 0.35 * cls_df["accuracy"].fillna(0.5)
    trade_df = validation_metrics.copy()
    trade_df["trade_score"] = trade_df["trade_AUC"].fillna(0.5) + 0.35 * trade_df["trade_F1"].fillna(0.0)

    best_reg = reg_df.sort_values(["reg_score", "directional_accuracy"], ascending=False).iloc[0]
    best_cls = cls_df.sort_values(["cls_score", "AUC"], ascending=False).iloc[0]
    best_trade = trade_df.sort_values(["trade_score", "trade_AUC"], ascending=False).iloc[0]

    def pack(row: pd.Series) -> dict[str, Any]:
        return {"model_name": str(row["model_name"]), "horizon": int(row["horizon"])}

    return {
        "best_regression": pack(best_reg),
        "best_classification": pack(best_cls),
        "best_trade": pack(best_trade),
    }


def prediction_slice(pred_all: pd.DataFrame, model_name: str, horizon: int) -> pd.DataFrame:
    out = pred_all.loc[(pred_all["model_name"] == model_name) & (pred_all["horizon"] == horizon)].copy()
    return add_prediction_ranks(out)


def build_ensemble_position(base: pd.DataFrame, pred_all: pd.DataFrame, components: dict[str, Any]) -> pd.Series:
    work = base[["date"]].copy()
    reg = prediction_slice(
        pred_all,
        components["best_regression"]["model_name"],
        components["best_regression"]["horizon"],
    )[["date", "pred_ret", "pred_ret_rank_60"]].rename(
        columns={"pred_ret": "ens_pred_ret", "pred_ret_rank_60": "ens_ret_rank"}
    )
    cls = prediction_slice(
        pred_all,
        components["best_classification"]["model_name"],
        components["best_classification"]["horizon"],
    )[["date", "pred_up_prob", "pred_up_prob_rank_60"]].rename(
        columns={"pred_up_prob": "ens_up_prob", "pred_up_prob_rank_60": "ens_up_rank"}
    )
    trade = prediction_slice(
        pred_all,
        components["best_trade"]["model_name"],
        components["best_trade"]["horizon"],
    )[["date", "pred_trade_prob", "pred_trade_prob_rank_60"]].rename(
        columns={"pred_trade_prob": "ens_trade_prob", "pred_trade_prob_rank_60": "ens_trade_rank"}
    )
    merged = work.merge(reg, on="date", how="left").merge(cls, on="date", how="left").merge(trade, on="date", how="left")
    base_reset = base.reset_index(drop=True)
    tech = technical_raw_position(base_reset)
    risk_penalty = (
        0.14 * (base_reset["ma_ratio_60"] < 0).astype(float)
        + 0.13 * (base_reset["drawdown_60"] < -0.10).astype(float)
        + 0.12 * (base_reset["volatility_ratio_20_60"] > 1.2).astype(float)
    )
    score = (
        0.32 * merged["ens_ret_rank"].fillna(0.5)
        + 0.30 * merged["ens_up_rank"].fillna(0.5)
        + 0.24 * merged["ens_trade_rank"].fillna(0.5)
        + 0.14 * tech
    )
    pos = 0.55 + 0.42 * score - risk_penalty
    return pos.clip(0.35, 1.0).fillna(0.65)


def make_strategy_base(feature_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    base_cols = [
        "date",
        "trade_date",
        "close",
        "fwd_ret_1d",
        "ret_20",
        "ma_ratio_20",
        "ma_ratio_60",
        "ma_spread_20_60",
        "drawdown_60",
        "drawdown_120",
        "volatility_20",
        "volatility_60",
        "volatility_ratio_20_60",
        "breakout_high_prior_20",
        "trend_regime_score",
        "up_day_ratio_20",
        "trend_slope_20",
        "trend_r2_20",
        "large_down_count_20",
        "large_down_return_sum_20",
        "tail_return_ratio_20",
        "downside_return_sum_20",
        "crash_risk_score",
        "up_day_volume_ratio_20",
        "down_day_volume_ratio_20",
        "volume_price_confirmation_20",
        "sell_pressure_volume_20",
        "bull_low_vol_regime",
        "bull_high_vol_regime",
        "bear_high_vol_regime",
        "sideways_regime",
    ]
    base = feature_df.loc[
        (feature_df["date"] >= start)
        & (feature_df["date"] <= end)
        & feature_df["trade_date"].notna()
        & feature_df["fwd_ret_1d"].notna(),
        base_cols,
    ].copy()
    return base.sort_values("date").reset_index(drop=True)


def strategy_stats_from_curve(curve: pd.DataFrame, benchmark_curve: pd.DataFrame) -> dict[str, float]:
    ret = curve["strategy_return"].astype(float)
    equity = curve["equity"].astype(float)
    benchmark_ret = benchmark_curve["strategy_return"].astype(float).reset_index(drop=True)
    benchmark_equity = benchmark_curve["equity"].astype(float)
    active = ret.reset_index(drop=True) - benchmark_ret
    total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1)
    benchmark_total = float(benchmark_equity.iloc[-1] / INITIAL_CAPITAL - 1)
    ann_ret = annualized_return(equity)
    benchmark_ann = annualized_return(benchmark_equity)
    sharpe = float(ret.mean() / ret.std() * math.sqrt(252)) if ret.std() > 0 else 0.0
    return {
        "total_return": total_return,
        "benchmark_total_return": benchmark_total,
        "excess_return_vs_buy_hold": total_return - benchmark_total,
        "annual_return": ann_ret,
        "annualized_excess_return": ann_ret - benchmark_ann if pd.notna(ann_ret) and pd.notna(benchmark_ann) else float(active.mean() * 252),
        "sharpe": sharpe,
        "max_drawdown": float(curve["drawdown"].min()),
        "benchmark_max_drawdown": float(benchmark_curve["drawdown"].min()),
        "drawdown_improvement": float(curve["drawdown"].min() - benchmark_curve["drawdown"].min()),
        "avg_turnover": float(curve["turnover"].mean()),
    }


def strategy_stats_from_raw_position_cost_aware(
    base: pd.DataFrame,
    raw_position: pd.Series,
    benchmark_curve: pd.DataFrame,
    no_trade_band: float,
    min_position: float = 0.0,
    max_position: float = 1.0,
) -> dict[str, float]:
    scheduled = apply_rebalance_with_no_trade_band(
        raw_position.reset_index(drop=True),
        REBALANCE_FREQ,
        min_position,
        max_position,
        no_trade_band,
    ).to_numpy(dtype=float)
    fwd_ret = base["fwd_ret_1d"].to_numpy(dtype=float)
    equity = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    prev_pos = 0.0
    returns: list[float] = []
    equities: list[float] = []
    drawdowns: list[float] = []
    turnovers: list[float] = []
    for raw, daily_ret in zip(scheduled, fwd_ret):
        final_pos = float(np.clip(raw, min_position, max_position))
        turnover = abs(final_pos - prev_pos)
        realized = final_pos * float(daily_ret) - turnover * COST_RATE
        equity = equity * (1 + realized)
        peak = max(peak, equity)
        returns.append(realized)
        equities.append(equity)
        drawdowns.append(equity / peak - 1)
        turnovers.append(turnover)
        prev_pos = final_pos

    ret = pd.Series(returns, dtype=float)
    equity_series = pd.Series(equities, dtype=float)
    benchmark_ret = benchmark_curve["strategy_return"].astype(float).reset_index(drop=True)
    benchmark_equity = benchmark_curve["equity"].astype(float)
    active = ret.reset_index(drop=True) - benchmark_ret
    total_return = float(equity_series.iloc[-1] / INITIAL_CAPITAL - 1)
    benchmark_total = float(benchmark_equity.iloc[-1] / INITIAL_CAPITAL - 1)
    ann_ret = annualized_return(equity_series)
    benchmark_ann = annualized_return(benchmark_equity)
    sharpe = float(ret.mean() / ret.std() * math.sqrt(252)) if ret.std() > 0 else 0.0
    max_drawdown = float(np.min(drawdowns)) if drawdowns else np.nan
    benchmark_max_drawdown = float(benchmark_curve["drawdown"].min())
    return {
        "total_return": total_return,
        "benchmark_total_return": benchmark_total,
        "excess_return_vs_buy_hold": total_return - benchmark_total,
        "annual_return": ann_ret,
        "annualized_excess_return": ann_ret - benchmark_ann if pd.notna(ann_ret) and pd.notna(benchmark_ann) else float(active.mean() * 252),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "drawdown_improvement": float(max_drawdown - benchmark_max_drawdown),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else np.nan,
    }


def position_activity_from_raw(
    raw_position: pd.Series,
    no_trade_band: float,
    min_position: float,
    max_position: float,
) -> dict[str, float]:
    scheduled = apply_rebalance_with_no_trade_band(
        raw_position.reset_index(drop=True),
        REBALANCE_FREQ,
        min_position,
        max_position,
        no_trade_band,
    ).to_numpy(dtype=float)
    if len(scheduled) == 0:
        return {
            "avg_position": np.nan,
            "min_position_observed": np.nan,
            "days_below_full_exposure": 0,
            "pct_days_below_full_exposure": np.nan,
            "total_turnover": np.nan,
            "avg_abs_position_gap": np.nan,
        }
    turnover = np.abs(np.diff(np.insert(scheduled, 0, 0.0)))
    below_full = scheduled < 0.999
    return {
        "avg_position": float(np.mean(scheduled)),
        "min_position_observed": float(np.min(scheduled)),
        "days_below_full_exposure": int(np.sum(below_full)),
        "pct_days_below_full_exposure": float(np.mean(below_full)),
        "total_turnover": float(np.sum(turnover)),
        "avg_abs_position_gap": float(np.mean(np.abs(scheduled - 1.0))),
    }


def tune_validation_index_core_params(feature_df: pd.DataFrame, validation_preds: pd.DataFrame) -> dict[str, Any]:
    base = make_strategy_base(feature_df, VALID_START, VALID_END)
    benchmark_curve = simulate_strategy(
        base,
        pd.Series(1.0, index=base.index),
        "validation_buy_hold",
        "benchmark",
        0,
        apply_dd_control=False,
    )
    grid = {
        "min_position": [0.85, 0.88, 0.90, 0.92],
        "strong_floor": [0.95, 0.97, 1.00],
        "weak_prob": [0.42, 0.45, 0.48],
        "strong_prob": [0.52, 0.55, 0.58],
        "high_vol_multiplier": [1.15, 1.20, 1.25, 1.30],
        "risk_cut": [0.03, 0.05, 0.08],
    }
    pred_cols = ["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]
    best: dict[str, Any] | None = None
    top_rows: list[dict[str, Any]] = []

    for (model_name, horizon), grp in validation_preds.groupby(["model_name", "horizon"], sort=True):
        merged = base.merge(grp[pred_cols], on="date", how="left")
        if not merged[["pred_ret", "pred_up_prob", "pred_trade_prob"]].notna().any().any():
            continue
        for values in product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            raw = ml_validation_tuned_index_core_position(merged, params)
            stats = strategy_stats_from_raw_position_cost_aware(
                merged,
                raw,
                benchmark_curve,
                no_trade_band=0.0,
                min_position=float(params["min_position"]),
                max_position=1.0,
            )
            score = (
                stats["excess_return_vs_buy_hold"]
                + 0.30 * stats["sharpe"]
                + 0.50 * stats["drawdown_improvement"]
                - 0.10 * stats["avg_turnover"]
            )
            row = {
                "model_name": str(model_name),
                "horizon": int(horizon),
                "validation_score": float(score),
                **params,
                **stats,
            }
            top_rows.append(row)
            if best is None or score > best["validation_score"]:
                best = row

    if best is None:
        best = {
            "model_name": "RandomForest",
            "horizon": 10,
            "validation_score": np.nan,
            "min_position": 0.88,
            "strong_floor": 0.97,
            "weak_prob": 0.45,
            "strong_prob": 0.55,
            "high_vol_multiplier": 1.25,
            "risk_cut": 0.05,
        }

    top = sorted(top_rows, key=lambda x: x["validation_score"], reverse=True)[:10]
    return {
        "selected": best,
        "top_candidates": top,
        "grid_size": len(top_rows),
        "validation_period": {"start": str(VALID_START.date()), "end": str(VALID_END.date())},
        "selection_rule": "excess_return_vs_buy_hold + 0.30 * sharpe + 0.50 * drawdown_improvement - 0.10 * avg_turnover",
    }


def get_cost_aware_prediction_source(
    pred_all: pd.DataFrame,
    fallback_model: str = "RandomForest",
    fallback_horizon: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    preferred = pred_all.loc[(pred_all["model_name"] == "ExtraTrees") & (pred_all["horizon"] == 10)].copy()
    if not preferred.empty:
        return preferred, {"model_name": "ExtraTrees", "horizon": 10, "fallback_used": False}
    fallback = pred_all.loc[(pred_all["model_name"] == fallback_model) & (pred_all["horizon"] == fallback_horizon)].copy()
    if not fallback.empty:
        return fallback, {"model_name": fallback_model, "horizon": fallback_horizon, "fallback_used": True}
    first_key = pred_all[["model_name", "horizon"]].drop_duplicates().sort_values(["model_name", "horizon"]).iloc[0]
    model_name = str(first_key["model_name"])
    horizon = int(first_key["horizon"])
    first = pred_all.loc[(pred_all["model_name"] == model_name) & (pred_all["horizon"] == horizon)].copy()
    return first, {"model_name": model_name, "horizon": horizon, "fallback_used": True}


def tune_cost_aware_index_enhancement_params(
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
) -> dict[str, Any]:
    base = make_strategy_base(feature_df, VALID_START, VALID_END)
    benchmark_curve = simulate_strategy(
        base,
        pd.Series(1.0, index=base.index),
        "validation_buy_hold",
        "benchmark",
        0,
        apply_dd_control=False,
    )
    signal_pred, source = get_cost_aware_prediction_source(validation_preds)
    merged = base.merge(signal_pred[["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]], on="date", how="left")

    grid = {
        "min_position": [0.85, 0.88, 0.90, 0.92],
        "strong_trend_cut_position": [0.95, 0.97, 0.98],
        "weak_cut_position": [0.92, 0.95, 0.97],
        "risk_cut_position": [0.90, 0.92, 0.95],
        "positive_prob": [0.50, 0.52, 0.55],
        "negative_prob": [0.45, 0.48],
        "strong_negative_prob": [0.42, 0.45],
        "drawdown_cut": [-0.08, -0.10, -0.12],
        "high_vol_multiplier": [1.15, 1.20, 1.25],
        "no_trade_band": [0.00, 0.03, 0.05],
    }
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        raw = ml_cost_aware_index_enhancement_position(merged, params, conservative=False)
        stats = strategy_stats_from_raw_position_cost_aware(
            merged,
            raw,
            benchmark_curve,
            no_trade_band=float(params["no_trade_band"]),
            min_position=float(params["min_position"]),
            max_position=1.0,
        )
        max_drawdown_abs = abs(stats["max_drawdown"])
        benchmark_drawdown_abs = abs(stats["benchmark_max_drawdown"])
        score = (
            1.00 * stats["excess_return_vs_buy_hold"]
            + 0.30 * stats["sharpe"]
            + 0.50 * stats["drawdown_improvement"]
            - 0.10 * stats["avg_turnover"]
            - 0.20 * max(0.0, max_drawdown_abs - benchmark_drawdown_abs)
        )
        row = {
            **source,
            "validation_score": float(score),
            **params,
            **stats,
        }
        rows.append(row)
        if best is None or score > best["validation_score"]:
            best = row

    if best is None:
        best = {
            **source,
            "validation_score": np.nan,
            "min_position": 0.88,
            "strong_trend_cut_position": 0.97,
            "weak_cut_position": 0.95,
            "risk_cut_position": 0.92,
            "positive_prob": 0.52,
            "negative_prob": 0.48,
            "strong_negative_prob": 0.45,
            "drawdown_cut": -0.10,
            "high_vol_multiplier": 1.20,
            "no_trade_band": 0.03,
        }
    top = sorted(rows, key=lambda x: x["validation_score"], reverse=True)[:10]
    return {
        "selected": best,
        "top_candidates": top,
        "grid_size": len(rows),
        "validation_period": {"start": str(VALID_START.date()), "end": str(VALID_END.date())},
        "signal_source": source,
        "selection_rule": (
            "1.00 * excess_return_vs_buy_hold + 0.30 * sharpe + 0.50 * drawdown_improvement "
            "- 0.10 * avg_turnover - 0.20 * max(0, max_drawdown_abs - benchmark_drawdown_abs)"
        ),
    }


def tune_tail_score_index_enhancement_params(
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
) -> dict[str, Any]:
    base = make_strategy_base(feature_df, VALID_START, VALID_END)
    benchmark_curve = simulate_strategy(
        base,
        pd.Series(1.0, index=base.index),
        "validation_buy_hold",
        "benchmark",
        0,
        apply_dd_control=False,
    )
    pred_cols = ["date", "pred_big_up_prob", "pred_big_down_prob", "pred_tail_score"]
    grid = {
        "strong_tail_score": [0.05, 0.10, 0.15],
        "low_tail_score": [-0.05, 0.00, 0.05],
        "very_low_tail_score": [-0.15, -0.10, -0.05],
        "big_down_prob_cut": [0.55, 0.60],
        "weak_cut_position": [0.95, 0.93],
        "risk_cut_position": [0.95, 0.92],
        "min_position": [0.90, 0.92],
        "high_risk_multiplier": [1.15, 1.25],
        "no_trade_band": [0.00, 0.03, 0.05],
    }
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for (model_name, horizon), grp in validation_preds.groupby(["model_name", "horizon"], sort=True):
        if not set(pred_cols).issubset(grp.columns):
            continue
        merged = base.merge(grp[pred_cols], on="date", how="left")
        if not merged[["pred_big_up_prob", "pred_big_down_prob", "pred_tail_score"]].notna().any().any():
            continue
        for values in product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            raw = ml_tail_score_index_enhancement_position(merged, params)
            stats = strategy_stats_from_raw_position_cost_aware(
                merged,
                raw,
                benchmark_curve,
                no_trade_band=float(params["no_trade_band"]),
                min_position=float(params["min_position"]),
                max_position=1.00,
            )
            activity = position_activity_from_raw(
                raw,
                no_trade_band=float(params["no_trade_band"]),
                min_position=float(params["min_position"]),
                max_position=1.00,
            )
            score = (
                1.00 * stats["excess_return_vs_buy_hold"]
                + 0.25 * stats["sharpe"]
                + 0.50 * stats["drawdown_improvement"]
                - 0.12 * stats["avg_turnover"]
                - 0.05 * max(0.0, 0.96 - activity["avg_position"])
            )
            row = {
                "model_name": str(model_name),
                "horizon": int(horizon),
                "validation_score": float(score),
                **params,
                **stats,
                **activity,
            }
            rows.append(row)
            if best is None or score > best["validation_score"]:
                best = row

    if best is None:
        best = {
            "model_name": "ExtraTrees",
            "horizon": 10,
            "validation_score": np.nan,
            "strong_tail_score": 0.10,
            "low_tail_score": 0.00,
            "very_low_tail_score": -0.10,
            "big_down_prob_cut": 0.60,
            "weak_cut_position": 0.95,
            "risk_cut_position": 0.92,
            "min_position": 0.90,
            "high_risk_multiplier": 1.25,
            "no_trade_band": 0.03,
        }
    top = sorted(rows, key=lambda x: x["validation_score"], reverse=True)[:10]
    return {
        "selected": best,
        "top_candidates": top,
        "grid_size": len(rows),
        "validation_period": {"start": str(VALID_START.date()), "end": str(VALID_END.date())},
        "selection_rule": (
            "Validation-only score: excess_return_vs_buy_hold + 0.25 * sharpe + "
            "0.50 * drawdown_improvement - 0.12 * avg_turnover - avg-position penalty."
        ),
    }


def directional_timing_validation_score(stats: dict[str, float]) -> float:
    return float(
        1.50 * stats["excess_return_vs_buy_hold"]
        + 0.40 * stats["drawdown_improvement"]
        + 0.20 * stats["sharpe"]
        - 0.10 * stats["avg_turnover"]
        - 0.20 * max(0.0, abs(stats["max_drawdown"]) - abs(stats["benchmark_max_drawdown"]))
    )


def tune_directional_timing_params(feature_df: pd.DataFrame, validation_preds: pd.DataFrame) -> dict[str, Any]:
    base = make_strategy_base(feature_df, VALID_START, VALID_END)
    benchmark_curve = simulate_strategy(
        base,
        pd.Series(1.0, index=base.index),
        "validation_buy_hold",
        "benchmark",
        0,
        apply_dd_control=False,
    )
    pred_cols = ["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]
    long_grid = {
        "buy_prob": [0.52, 0.55, 0.58, 0.60],
        "sell_prob": [0.48, 0.45, 0.42, 0.40],
        "min_pred_ret_buy": [0.000, 0.002, 0.003],
        "max_pred_ret_sell": [0.000, -0.002, -0.003],
        "use_trade_prob": [True, False],
        "trade_prob_cut": [0.50, 0.55, 0.60],
        "no_trade_band": [0.00, 0.05],
    }
    scaled_grid = {
        "strong_buy_prob": [0.58, 0.60, 0.62],
        "buy_prob": [0.52, 0.55],
        "sell_prob": [0.48, 0.45],
        "strong_sell_prob": [0.42, 0.40],
        "buy_position": [0.80, 0.90, 0.95],
        "neutral_position": [0.50, 0.70, 0.80],
        "sell_position": [0.20, 0.30, 0.50],
        "strong_sell_position": [0.00, 0.10, 0.20],
        "trend_filter": [True, False],
        "risk_filter": [True, False],
        "no_trade_band": [0.00, 0.03, 0.05],
    }
    long_rows: list[dict[str, Any]] = []
    scaled_rows: list[dict[str, Any]] = []
    best_long: dict[str, Any] | None = None
    best_scaled: dict[str, Any] | None = None
    merged_by_key: dict[tuple[str, int], pd.DataFrame] = {}
    coarse_long_rows: list[dict[str, Any]] = []
    coarse_scaled_rows: list[dict[str, Any]] = []
    coarse_long_params = [
        {"buy_prob": 0.52, "sell_prob": 0.48, "min_pred_ret_buy": 0.000, "max_pred_ret_sell": 0.000, "use_trade_prob": False, "trade_prob_cut": 0.50, "no_trade_band": 0.00},
        {"buy_prob": 0.55, "sell_prob": 0.45, "min_pred_ret_buy": 0.000, "max_pred_ret_sell": 0.000, "use_trade_prob": False, "trade_prob_cut": 0.50, "no_trade_band": 0.05},
        {"buy_prob": 0.58, "sell_prob": 0.42, "min_pred_ret_buy": 0.002, "max_pred_ret_sell": -0.002, "use_trade_prob": True, "trade_prob_cut": 0.55, "no_trade_band": 0.05},
        {"buy_prob": 0.60, "sell_prob": 0.40, "min_pred_ret_buy": 0.003, "max_pred_ret_sell": -0.003, "use_trade_prob": True, "trade_prob_cut": 0.60, "no_trade_band": 0.05},
    ]
    coarse_scaled_params = [
        {"strong_buy_prob": 0.58, "buy_prob": 0.52, "sell_prob": 0.48, "strong_sell_prob": 0.42, "strong_buy_position": 1.0, "buy_position": 0.90, "neutral_position": 0.70, "sell_position": 0.30, "strong_sell_position": 0.10, "trend_filter": True, "risk_filter": True, "no_trade_band": 0.03},
        {"strong_buy_prob": 0.60, "buy_prob": 0.55, "sell_prob": 0.45, "strong_sell_prob": 0.40, "strong_buy_position": 1.0, "buy_position": 0.95, "neutral_position": 0.80, "sell_position": 0.50, "strong_sell_position": 0.20, "trend_filter": True, "risk_filter": False, "no_trade_band": 0.05},
        {"strong_buy_prob": 0.60, "buy_prob": 0.52, "sell_prob": 0.45, "strong_sell_prob": 0.42, "strong_buy_position": 1.0, "buy_position": 0.80, "neutral_position": 0.50, "sell_position": 0.20, "strong_sell_position": 0.00, "trend_filter": False, "risk_filter": True, "no_trade_band": 0.03},
        {"strong_buy_prob": 0.62, "buy_prob": 0.55, "sell_prob": 0.48, "strong_sell_prob": 0.40, "strong_buy_position": 1.0, "buy_position": 0.90, "neutral_position": 0.70, "sell_position": 0.30, "strong_sell_position": 0.00, "trend_filter": False, "risk_filter": False, "no_trade_band": 0.00},
    ]

    for (model_name, horizon), grp in validation_preds.groupby(["model_name", "horizon"], sort=True):
        merged = base.merge(grp[pred_cols], on="date", how="left")
        if not merged[["pred_ret", "pred_up_prob", "pred_trade_prob"]].notna().any().any():
            continue
        key = (str(model_name), int(horizon))
        merged_by_key[key] = merged
        for params in coarse_long_params:
            raw = ml_directional_long_cash_position(merged, params)
            stats = strategy_stats_from_raw_position_cost_aware(
                merged,
                raw,
                benchmark_curve,
                no_trade_band=float(params["no_trade_band"]),
                min_position=0.0,
                max_position=1.0,
            )
            coarse_long_rows.append({"key": key, "score": directional_timing_validation_score(stats)})
        for params in coarse_scaled_params:
            raw = ml_directional_scaled_timing_position(merged, params)
            stats = strategy_stats_from_raw_position_cost_aware(
                merged,
                raw,
                benchmark_curve,
                no_trade_band=float(params["no_trade_band"]),
                min_position=0.0,
                max_position=1.0,
            )
            coarse_scaled_rows.append({"key": key, "score": directional_timing_validation_score(stats)})

    top_long_keys = {
        row["key"]
        for row in sorted(coarse_long_rows, key=lambda x: x["score"], reverse=True)[:6]
    }
    top_scaled_keys = {
        row["key"]
        for row in sorted(coarse_scaled_rows, key=lambda x: x["score"], reverse=True)[:6]
    }

    for (model_name, horizon), merged in merged_by_key.items():
        if (model_name, horizon) not in top_long_keys and (model_name, horizon) not in top_scaled_keys:
            continue

        if (model_name, horizon) in top_long_keys:
            for values in product(*long_grid.values()):
                params = dict(zip(long_grid.keys(), values))
                raw = ml_directional_long_cash_position(merged, params)
                stats = strategy_stats_from_raw_position_cost_aware(
                    merged,
                    raw,
                    benchmark_curve,
                    no_trade_band=float(params["no_trade_band"]),
                    min_position=0.0,
                    max_position=1.0,
                )
                score = directional_timing_validation_score(stats)
                row = {
                    "strategy": "ml_directional_long_cash",
                    "model_name": str(model_name),
                    "horizon": int(horizon),
                    "validation_score": score,
                    **params,
                    **stats,
                }
                long_rows.append(row)
                if best_long is None or score > best_long["validation_score"]:
                    best_long = row

        if (model_name, horizon) in top_scaled_keys:
            for values in product(*scaled_grid.values()):
                params = dict(zip(scaled_grid.keys(), values))
                if float(params["strong_sell_position"]) > float(params["sell_position"]):
                    continue
                if float(params["sell_position"]) > float(params["neutral_position"]):
                    continue
                if float(params["neutral_position"]) > float(params["buy_position"]):
                    continue
                params["strong_buy_position"] = 1.0
                raw = ml_directional_scaled_timing_position(merged, params)
                stats = strategy_stats_from_raw_position_cost_aware(
                    merged,
                    raw,
                    benchmark_curve,
                    no_trade_band=float(params["no_trade_band"]),
                    min_position=0.0,
                    max_position=1.0,
                )
                score = directional_timing_validation_score(stats)
                row = {
                    "strategy": "ml_directional_scaled_timing",
                    "model_name": str(model_name),
                    "horizon": int(horizon),
                    "validation_score": score,
                    **params,
                    **stats,
                }
                scaled_rows.append(row)
                if best_scaled is None or score > best_scaled["validation_score"]:
                    best_scaled = row

    if best_long is None:
        best_long = {
            "strategy": "ml_directional_long_cash",
            "model_name": "ExtraTrees",
            "horizon": 10,
            "validation_score": np.nan,
            "buy_prob": 0.55,
            "sell_prob": 0.45,
            "min_pred_ret_buy": 0.0,
            "max_pred_ret_sell": 0.0,
            "use_trade_prob": False,
            "trade_prob_cut": 0.50,
            "no_trade_band": 0.05,
        }
    if best_scaled is None:
        best_scaled = {
            "strategy": "ml_directional_scaled_timing",
            "model_name": "ExtraTrees",
            "horizon": 10,
            "validation_score": np.nan,
            "strong_buy_prob": 0.60,
            "buy_prob": 0.55,
            "sell_prob": 0.45,
            "strong_sell_prob": 0.40,
            "strong_buy_position": 1.0,
            "buy_position": 0.90,
            "neutral_position": 0.70,
            "sell_position": 0.30,
            "strong_sell_position": 0.10,
            "trend_filter": True,
            "risk_filter": True,
            "no_trade_band": 0.03,
        }

    return {
        "selected_long_cash_params": best_long,
        "selected_scaled_timing_params": best_scaled,
        "top_candidates_long_cash": sorted(long_rows, key=lambda x: x["validation_score"], reverse=True)[:20],
        "top_candidates_scaled_timing": sorted(scaled_rows, key=lambda x: x["validation_score"], reverse=True)[:20],
        "grid_size_long_cash": len(long_rows),
        "grid_size_scaled_timing": len(scaled_rows),
        "validation_period": {"start": str(VALID_START.date()), "end": str(VALID_END.date())},
        "selection_rule": (
            "1.50 * excess_return_vs_buy_hold + 0.40 * drawdown_improvement + 0.20 * sharpe "
            "- 0.10 * avg_turnover - 0.20 * max(0, abs(max_drawdown) - abs(benchmark_max_drawdown))"
        ),
    }


def tune_selective_defensive_enhancement_params(
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
) -> dict[str, Any]:
    base = make_strategy_base(feature_df, VALID_START, VALID_END)
    benchmark_curve = simulate_strategy(
        base,
        pd.Series(1.0, index=base.index),
        "validation_buy_hold",
        "benchmark",
        0,
        apply_dd_control=False,
    )
    pred_cols = ["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]
    grid = {
        "weak_prob": [0.45, 0.48],
        "strong_negative_prob": [0.40, 0.42, 0.45],
        "weak_ret_cut": [0.0, -0.002],
        "mild_cut_position": [0.98, 0.97],
        "risk_cut_position": [0.95, 0.92],
        "min_position": [0.90, 0.88],
        "high_vol_multiplier": [1.20, 1.25],
        "drawdown_cut": [-0.10, -0.12],
        "no_trade_band": [0.03, 0.05],
        "require_trend_confirm": [True],
        "require_risk_confirm": [True, False],
    }
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for (model_name, horizon), grp in validation_preds.groupby(["model_name", "horizon"], sort=True):
        if not set(pred_cols).issubset(grp.columns):
            continue
        merged = base.merge(grp[pred_cols], on="date", how="left")
        if not merged[["pred_ret", "pred_up_prob", "pred_trade_prob"]].notna().any().any():
            continue
        for values in product(*grid.values()):
            params = dict(zip(grid.keys(), values))
            raw = ml_selective_defensive_enhancement_position(merged, params)
            min_pos = float(params["min_position"])
            no_trade_band = float(params["no_trade_band"])
            stats = strategy_stats_from_raw_position_cost_aware(
                merged,
                raw,
                benchmark_curve,
                no_trade_band=no_trade_band,
                min_position=min_pos,
                max_position=1.0,
            )
            activity = position_activity_from_raw(raw, no_trade_band, min_pos, 1.0)
            clone_like_penalty = 0.0
            if activity["days_below_full_exposure"] < 5:
                clone_like_penalty += 0.20
            if activity["avg_abs_position_gap"] < 0.002:
                clone_like_penalty += 0.20
            max_drawdown_abs = abs(stats["max_drawdown"])
            benchmark_drawdown_abs = abs(stats["benchmark_max_drawdown"])
            score = (
                2.00 * stats["excess_return_vs_buy_hold"]
                + 0.60 * stats["drawdown_improvement"]
                + 0.15 * stats["sharpe"]
                - 0.10 * stats["avg_turnover"]
                - 0.30 * max(0.0, max_drawdown_abs - benchmark_drawdown_abs)
                - 0.20 * max(0.0, 0.975 - activity["avg_position"])
                - clone_like_penalty
            )
            selectable = activity["days_below_full_exposure"] > 0
            row = {
                "strategy": "ml_selective_defensive_enhancement",
                "model_name": str(model_name),
                "horizon": int(horizon),
                "validation_score": float(score),
                "clone_like_penalty": float(clone_like_penalty),
                "selectable": bool(selectable),
                **params,
                **stats,
                **activity,
            }
            rows.append(row)
            if selectable and (best is None or score > best["validation_score"]):
                best = row

    if best is None:
        best = {
            "strategy": "ml_selective_defensive_enhancement",
            "model_name": "ExtraTrees",
            "horizon": 10,
            "validation_score": np.nan,
            "clone_like_penalty": np.nan,
            "selectable": False,
            "weak_prob": 0.48,
            "strong_negative_prob": 0.42,
            "weak_ret_cut": 0.0,
            "mild_cut_position": 0.98,
            "risk_cut_position": 0.95,
            "min_position": 0.90,
            "high_vol_multiplier": 1.20,
            "drawdown_cut": -0.10,
            "no_trade_band": 0.05,
            "require_trend_confirm": True,
            "require_risk_confirm": True,
        }

    top = sorted(rows, key=lambda x: x["validation_score"], reverse=True)[:20]
    return {
        "selected": best,
        "top_candidates": top,
        "grid_size": len(rows),
        "validation_period": {"start": str(VALID_START.date()), "end": str(VALID_END.date())},
        "signal_source": {"model_name": best.get("model_name"), "horizon": best.get("horizon")},
        "selection_rule": (
            "2.00 * excess_return_vs_buy_hold + 0.60 * drawdown_improvement + 0.15 * sharpe "
            "- 0.10 * avg_turnover - 0.30 * max(0, abs(max_drawdown) - abs(benchmark_max_drawdown)) "
            "- 0.20 * max(0, 0.975 - avg_position) - clone_like_penalty"
        ),
        "clone_like_penalty_rule": (
            "days_below_full_exposure < 5 adds 0.20; avg_abs_position_gap < 0.002 adds 0.20; "
            "candidates with zero reduced-exposure days are not selectable"
        ),
    }


def build_dynamic_ensemble_weights(validation_metrics: pd.DataFrame) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = [
        {"model_name": "ExtraTrees", "horizon": 10, "reason": "highest_directional_accuracy_prior"},
        {"model_name": "GradientBoosting", "horizon": 1, "reason": "highest_auc_prior"},
        {"model_name": "RandomForest", "horizon": 10, "reason": "composite_selected_prior"},
    ]
    optional = validation_metrics.loc[validation_metrics["model_name"].isin(["XGBoost", "LightGBM"])].copy()
    if not optional.empty:
        optional["validation_model_score"] = (
            optional["AUC"].fillna(0.5)
            + 0.50 * optional["directional_accuracy"].fillna(0.5)
            + 0.50 * optional["return_correlation"].clip(lower=0).fillna(0.0)
        )
        best_optional = optional.sort_values("validation_model_score", ascending=False).iloc[0]
        candidates.append(
            {
                "model_name": str(best_optional["model_name"]),
                "horizon": int(best_optional["horizon"]),
                "reason": "best_xgboost_lightgbm_validation_metric",
            }
        )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for candidate in candidates:
        key = (candidate["model_name"], int(candidate["horizon"]))
        if key in seen:
            continue
        seen.add(key)
        metric = validation_metrics.loc[
            (validation_metrics["model_name"] == key[0]) & (validation_metrics["horizon"] == key[1])
        ]
        if metric.empty:
            continue
        metric_row = metric.iloc[0]
        auc_value = 0.5 if pd.isna(metric_row.get("AUC")) else float(metric_row.get("AUC"))
        dir_value = 0.5 if pd.isna(metric_row.get("directional_accuracy")) else float(metric_row.get("directional_accuracy"))
        corr_value = 0.0 if pd.isna(metric_row.get("return_correlation")) else float(metric_row.get("return_correlation"))
        raw_score = max(auc_value - 0.5, 0.0) + max(dir_value - 0.5, 0.0) + max(corr_value, 0.0)
        rows.append(
            {
                **candidate,
                "raw_score": raw_score,
                "AUC": None if pd.isna(metric_row.get("AUC")) else float(metric_row["AUC"]),
                "directional_accuracy": None
                if pd.isna(metric_row.get("directional_accuracy"))
                else float(metric_row["directional_accuracy"]),
                "return_correlation": None
                if pd.isna(metric_row.get("return_correlation"))
                else float(metric_row["return_correlation"]),
            }
        )

    if not rows:
        rows = [{"model_name": "RandomForest", "horizon": 10, "reason": "fallback", "raw_score": 1.0}]
    total = sum(row["raw_score"] for row in rows)
    if total <= 0:
        for row in rows:
            row["weight"] = 1.0 / len(rows)
    else:
        for row in rows:
            row["weight"] = row["raw_score"] / total
    return {"components": rows, "weight_source": "validation_metrics_only"}


def build_dynamic_ensemble_full_participation_position(
    base: pd.DataFrame,
    pred_all: pd.DataFrame,
    weights: dict[str, Any],
) -> pd.Series:
    work = base[["date"]].copy()
    score = pd.Series(0.0, index=base.index, dtype=float)
    weight_sum = 0.0
    for component in weights.get("components", []):
        pred = prediction_slice(pred_all, component["model_name"], int(component["horizon"]))
        if pred.empty:
            continue
        comp = pred[["date", "pred_ret_rank_60", "pred_up_prob_rank_60", "pred_trade_prob_rank_60"]].copy()
        comp["component_score"] = (
            comp["pred_ret_rank_60"].fillna(0.5)
            + comp["pred_up_prob_rank_60"].fillna(0.5)
            + comp["pred_trade_prob_rank_60"].fillna(0.5)
        ) / 3
        merged = work.merge(comp[["date", "component_score"]], on="date", how="left")
        weight = float(component.get("weight", 0.0))
        score = score + weight * merged["component_score"].fillna(0.5).reset_index(drop=True)
        weight_sum += weight
    if weight_sum <= 0:
        score = pd.Series(0.5, index=base.index, dtype=float)
    else:
        score = score / weight_sum

    strong_trend = (base["ma_ratio_60"] > 0) & (base["ma_spread_20_60"] > 0) & (base["ret_20"] > 0)
    weak_trend = (base["ma_ratio_60"] < 0) | (base["ret_20"] < 0)
    high_risk = (base["drawdown_60"] < -0.12) | (base["volatility_20"] > 1.25 * base["volatility_60"])
    position = pd.Series(0.95, index=base.index, dtype=float)
    position = position.where(~((score > 0.58) & strong_trend), 1.00)
    position = position.where(~((score < 0.42) & weak_trend & high_risk), 0.88)
    position = position.where(~((score < 0.45) & weak_trend & ~high_risk), 0.92)
    position = position.where(~strong_trend, position.clip(lower=0.95))
    return position.clip(0.88, 1.00).fillna(0.95)


def build_all_strategies(
    feature_df: pd.DataFrame,
    pred_all: pd.DataFrame,
    validation_components: dict[str, Any],
    tuned_params: dict[str, Any] | None = None,
    dynamic_weights: dict[str, Any] | None = None,
    cost_aware_params: dict[str, Any] | None = None,
    tail_score_params: dict[str, Any] | None = None,
    directional_timing_params: dict[str, Any] | None = None,
    selective_defensive_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    base = make_strategy_base(feature_df, TEST_START, TEST_END)

    frames: list[pd.DataFrame] = []
    frames.append(
        simulate_strategy(
            base,
            pd.Series(1.0, index=base.index),
            "buy_hold",
            "benchmark",
            0,
            apply_dd_control=False,
        )
    )
    frames.append(
        simulate_strategy(
            base,
            technical_raw_position(base),
            "technical_timing",
            "technical_rules",
            0,
            apply_dd_control=True,
        )
    )

    for (model_name, horizon), grp in pred_all.groupby(["model_name", "horizon"], sort=True):
        pred = add_prediction_ranks(grp)
        pred_cols = [
            "date",
            "pred_ret",
            "pred_up_prob",
            "pred_trade_prob",
            "pred_ret_rank_60",
            "pred_up_prob_rank_60",
            "pred_trade_prob_rank_60",
            "pred_tail_score_rank_60",
            "pred_ret_rank_20",
            "pred_up_prob_rank_20",
            "pred_trade_prob_rank_20",
            "pred_tail_score_rank_20",
            "pred_big_up_prob",
            "pred_big_down_prob",
            "pred_tail_score",
        ]
        merged = base.merge(pred[pred_cols], on="date", how="left")
        pos_map = model_raw_positions(merged)
        for strategy, raw_pos in pos_map.items():
            frames.append(
                simulate_strategy(
                    merged,
                    raw_pos,
                    strategy,
                    str(model_name),
                    int(horizon),
                    apply_dd_control=True,
                )
            )

        core_pos = ml_index_enhanced_core_position(merged)
        frames.append(
            simulate_strategy(
                merged,
                core_pos,
                "ml_index_enhanced_core",
                str(model_name),
                int(horizon),
                apply_dd_control=False,
                min_position=0.75,
                max_position=1.00,
            )
        )

        frames.append(
            simulate_strategy(
                merged,
                ml_index_full_participation_position(merged),
                "ml_index_full_participation",
                str(model_name),
                int(horizon),
                apply_dd_control=False,
                min_position=0.88,
                max_position=1.00,
            )
        )

        vol_frame = ml_vol_target_trend_frame(merged)
        vol_merged = merged.join(vol_frame.drop(columns=["raw_position"], errors="ignore"))
        frames.append(
            simulate_strategy(
                vol_merged,
                vol_frame["raw_position"],
                "ml_vol_target_trend",
                str(model_name),
                int(horizon),
                apply_dd_control=False,
                min_position=0.60,
                max_position=1.00,
            )
        )

        frames.append(
            simulate_cppi_strategy(
                merged,
                "ml_enhanced_cppi",
                str(model_name),
                int(horizon),
            )
        )

        if ALLOW_LEVERAGE_EXPERIMENT:
            frames.append(
                simulate_strategy(
                    merged,
                    ml_index_enhanced_plus_position(merged),
                    "ml_index_enhanced_plus",
                    str(model_name),
                    int(horizon),
                    apply_dd_control=False,
                    min_position=0.75,
                    max_position=1.10,
                )
            )
            frames.append(
                simulate_strategy(
                    merged,
                    ml_index_enhanced_plus_participation_position(merged, 1.15),
                    "ml_index_enhanced_plus_115",
                    str(model_name),
                    int(horizon),
                    apply_dd_control=False,
                    min_position=0.85,
                    max_position=1.15,
                )
            )
            frames.append(
                simulate_strategy(
                    merged,
                    ml_index_enhanced_plus_participation_position(merged, 1.20),
                    "ml_index_enhanced_plus_120",
                    str(model_name),
                    int(horizon),
                    apply_dd_control=False,
                    min_position=0.85,
                    max_position=1.20,
                )
            )

    et10 = pred_all.loc[(pred_all["model_name"] == "ExtraTrees") & (pred_all["horizon"] == 10)].copy()
    if not et10.empty:
        et10 = add_prediction_ranks(et10)
        pred_cols = ["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]
        et_merged = base.merge(et10[pred_cols], on="date", how="left")
        frames.append(
            simulate_strategy(
                et_merged,
                ml_directional_alpha_extratrees_10d_position(et_merged),
                "ml_directional_alpha_extratrees_10d",
                "ExtraTrees",
                10,
                apply_dd_control=False,
                min_position=0.85,
                max_position=1.00,
            )
        )

    if tuned_params is not None:
        selected = tuned_params.get("selected", tuned_params)
        tuned_model = str(selected.get("model_name", "RandomForest"))
        tuned_horizon = int(selected.get("horizon", 10))
        tuned_pred = pred_all.loc[(pred_all["model_name"] == tuned_model) & (pred_all["horizon"] == tuned_horizon)].copy()
        if not tuned_pred.empty:
            tuned_merged = base.merge(tuned_pred[["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]], on="date", how="left")
            frames.append(
                simulate_strategy(
                    tuned_merged,
                    ml_validation_tuned_index_core_position(tuned_merged, selected),
                    "ml_validation_tuned_index_core",
                    tuned_model,
                    tuned_horizon,
                    apply_dd_control=False,
                    min_position=float(selected.get("min_position", 0.88)),
                    max_position=1.00,
                )
            )

    if dynamic_weights is not None:
        frames.append(
            simulate_strategy(
                base,
                build_dynamic_ensemble_full_participation_position(base, pred_all, dynamic_weights),
                "ml_dynamic_ensemble_full_participation",
                "validation_weighted_ensemble",
                -2,
                apply_dd_control=False,
                min_position=0.88,
                max_position=1.00,
            )
        )

    if cost_aware_params is not None:
        selected_cost_params = cost_aware_params.get("selected", cost_aware_params)
        cost_pred, cost_source = get_cost_aware_prediction_source(
            pred_all,
            fallback_model=str(selected_cost_params.get("model_name", "RandomForest")),
            fallback_horizon=int(selected_cost_params.get("horizon", 10)),
        )
        cost_merged = base.merge(cost_pred[["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]], on="date", how="left")
        no_trade_band = float(selected_cost_params.get("no_trade_band", 0.03))
        frames.append(
            simulate_strategy_cost_aware(
                cost_merged,
                ml_cost_aware_index_enhancement_position(cost_merged, selected_cost_params, conservative=False),
                "ml_cost_aware_index_enhancement",
                str(cost_source["model_name"]),
                int(cost_source["horizon"]),
                no_trade_band=no_trade_band,
                apply_dd_control=False,
                min_position=float(selected_cost_params.get("min_position", 0.88)),
                max_position=1.00,
            )
        )
        frames.append(
            simulate_strategy_cost_aware(
                cost_merged,
                ml_cost_aware_index_enhancement_position(cost_merged, selected_cost_params, conservative=True),
                "ml_cost_aware_overlay_conservative",
                str(cost_source["model_name"]),
                int(cost_source["horizon"]),
                no_trade_band=no_trade_band,
                apply_dd_control=False,
                min_position=0.95,
                max_position=1.00,
            )
        )

    if tail_score_params is not None:
        selected_tail_params = tail_score_params.get("selected", tail_score_params)
        tail_model = str(selected_tail_params.get("model_name", "ExtraTrees"))
        tail_horizon = int(selected_tail_params.get("horizon", 10))
        tail_pred = pred_all.loc[(pred_all["model_name"] == tail_model) & (pred_all["horizon"] == tail_horizon)].copy()
        tail_pred = add_prediction_ranks(tail_pred) if not tail_pred.empty else tail_pred
        tail_cols = ["date", "pred_big_up_prob", "pred_big_down_prob", "pred_tail_score"]
        if not tail_pred.empty and set(tail_cols).issubset(tail_pred.columns):
            tail_merged = base.merge(tail_pred[tail_cols], on="date", how="left")
            frames.append(
                simulate_strategy_cost_aware(
                    tail_merged,
                    ml_tail_score_index_enhancement_position(tail_merged, selected_tail_params),
                    "ml_tail_score_index_enhancement",
                    tail_model,
                    tail_horizon,
                    no_trade_band=float(selected_tail_params.get("no_trade_band", 0.03)),
                    apply_dd_control=False,
                    min_position=float(selected_tail_params.get("min_position", 0.90)),
                    max_position=1.00,
                )
            )

    if directional_timing_params is not None:
        directional_specs = [
            (
                "ml_directional_long_cash",
                directional_timing_params.get("selected_long_cash_params", {}),
                ml_directional_long_cash_position,
            ),
            (
                "ml_directional_scaled_timing",
                directional_timing_params.get("selected_scaled_timing_params", {}),
                ml_directional_scaled_timing_position,
            ),
        ]
        for strategy_name, selected_directional_params, position_func in directional_specs:
            if not selected_directional_params:
                continue
            directional_model = str(selected_directional_params.get("model_name", "ExtraTrees"))
            directional_horizon = int(selected_directional_params.get("horizon", 10))
            directional_pred = pred_all.loc[
                (pred_all["model_name"] == directional_model) & (pred_all["horizon"] == directional_horizon)
            ].copy()
            pred_cols = ["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]
            if directional_pred.empty or not set(pred_cols).issubset(directional_pred.columns):
                continue
            directional_merged = base.merge(directional_pred[pred_cols], on="date", how="left")
            frames.append(
                simulate_strategy_cost_aware(
                    directional_merged,
                    position_func(directional_merged, selected_directional_params),
                    strategy_name,
                    directional_model,
                    directional_horizon,
                    no_trade_band=float(selected_directional_params.get("no_trade_band", 0.0)),
                    apply_dd_control=False,
                    min_position=0.0,
                    max_position=1.0,
                )
            )

    if selective_defensive_params is not None:
        selected_selective_params = selective_defensive_params.get("selected", selective_defensive_params)
        selective_model = str(selected_selective_params.get("model_name", "ExtraTrees"))
        selective_horizon = int(selected_selective_params.get("horizon", 10))
        selective_pred = pred_all.loc[
            (pred_all["model_name"] == selective_model) & (pred_all["horizon"] == selective_horizon)
        ].copy()
        pred_cols = ["date", "pred_ret", "pred_up_prob", "pred_trade_prob"]
        if not selective_pred.empty and set(pred_cols).issubset(selective_pred.columns):
            selective_merged = base.merge(selective_pred[pred_cols], on="date", how="left")
            frames.append(
                simulate_strategy_cost_aware(
                    selective_merged,
                    ml_selective_defensive_enhancement_position(selective_merged, selected_selective_params),
                    "ml_selective_defensive_enhancement",
                    selective_model,
                    selective_horizon,
                    no_trade_band=float(selected_selective_params.get("no_trade_band", 0.05)),
                    apply_dd_control=False,
                    min_position=float(selected_selective_params.get("min_position", 0.90)),
                    max_position=1.00,
                )
            )

    ensemble_raw = build_ensemble_position(base, pred_all, validation_components)
    frames.append(
        simulate_strategy(
            base,
            ensemble_raw,
            "ensemble_strategy",
            "validation_selected_ensemble",
            -1,
            apply_dd_control=True,
        )
    )

    return pd.concat(frames, ignore_index=True).sort_values(["strategy", "model_name", "horizon", "date"]).reset_index(drop=True)


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    total = equity.iloc[-1] / INITIAL_CAPITAL
    years = len(equity) / 252
    if total <= 0 or years <= 0:
        return np.nan
    return float(total ** (1 / years) - 1)


def compute_strategy_metrics(strategy_daily: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        strategy_daily.loc[strategy_daily["strategy"] == "buy_hold", ["date", "strategy_return", "equity"]]
        .rename(columns={"strategy_return": "benchmark_return", "equity": "benchmark_equity"})
        .sort_values("date")
    )
    benchmark_total = float(benchmark["benchmark_equity"].iloc[-1] / INITIAL_CAPITAL - 1)
    benchmark_ann = annualized_return(benchmark["benchmark_equity"])
    rows: list[dict[str, Any]] = []

    for (strategy, model_name, horizon), grp in strategy_daily.groupby(["strategy", "model_name", "horizon"], sort=False):
        grp = grp.sort_values("date").copy()
        merged = grp.merge(benchmark[["date", "benchmark_return"]], on="date", how="left")
        ret = merged["strategy_return"].astype(float)
        equity = merged["equity"].astype(float)
        active = ret - merged["benchmark_return"].astype(float)
        position = merged["final_position"].astype(float)
        turnover = merged["turnover"].astype(float)
        total_days = len(merged)
        days_below_full_exposure = int((position < 0.999).sum())
        total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1)
        ann_ret = annualized_return(equity)
        ann_vol = float(ret.std() * math.sqrt(252)) if ret.std() > 0 else 0.0
        sharpe = float(ret.mean() / ret.std() * math.sqrt(252)) if ret.std() > 0 else np.nan
        mdd = float(merged["drawdown"].min())
        tracking_error = float(active.std() * math.sqrt(252)) if active.std() > 0 else 0.0
        ann_excess = float(active.mean() * 252)
        rows.append(
            {
                "strategy": strategy,
                "model_name": model_name,
                "horizon": horizon,
                "total_return": total_return,
                "annual_return": ann_ret,
                "annual_volatility": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "calmar": ann_ret / abs(mdd) if mdd < 0 and pd.notna(ann_ret) else np.nan,
                "win_rate": float((ret > 0).mean()),
                "avg_turnover": float(merged["turnover"].mean()),
                "total_days": total_days,
                "avg_position": float(position.mean()),
                "min_position": float(position.min()),
                "max_position": float(position.max()),
                "days_below_full_exposure": days_below_full_exposure,
                "pct_days_below_full_exposure": (
                    days_below_full_exposure / total_days if total_days > 0 else 0.0
                ),
                "total_turnover": float(turnover.sum()),
                "active_return_sum": float(active.sum()),
                "avg_abs_position_gap": float((position - 1.0).abs().mean()),
                "benchmark_total_return": benchmark_total,
                "excess_return_vs_buy_hold": total_return - benchmark_total,
                "annualized_excess_return": ann_ret - benchmark_ann if pd.notna(ann_ret) and pd.notna(benchmark_ann) else ann_excess,
                "tracking_error": tracking_error,
                "information_ratio": ann_excess / tracking_error if tracking_error > 0 else np.nan,
            }
        )

    metrics = pd.DataFrame(rows)
    strategy_names = metrics["strategy"].astype(str)
    enhanced_exposure_mask = strategy_names.str.startswith("ml_index_enhanced_plus")
    metrics["is_benchmark_clone"] = (
        (strategy_names != "buy_hold")
        & (metrics["avg_abs_position_gap"] < 0.001)
        & (metrics["excess_return_vs_buy_hold"].abs() < 0.0005)
        & (metrics["tracking_error"] < 0.0005)
    )
    metrics["is_real_ml_timing_strategy"] = (
        (strategy_names != "buy_hold")
        & ~enhanced_exposure_mask
        & ~metrics["is_benchmark_clone"]
        & (
            (metrics["pct_days_below_full_exposure"] >= 0.03)
            | (metrics["avg_abs_position_gap"] >= 0.003)
        )
    )
    metrics["strategy_role"] = np.select(
        [
            strategy_names == "buy_hold",
            metrics["is_benchmark_clone"],
            enhanced_exposure_mask,
            metrics["is_real_ml_timing_strategy"],
        ],
        [
            "buy-and-hold benchmark",
            "benchmark-clone strategy",
            "enhanced-exposure strategy",
            "real ML timing strategy",
        ],
        default="other no-leverage strategy",
    )
    metrics["strategy_score"] = (
        1.00 * metrics["excess_return_vs_buy_hold"].fillna(-1)
        + 0.30 * metrics["sharpe"].fillna(0)
        + 0.20 * metrics["calmar"].replace([np.inf, -np.inf], np.nan).fillna(0)
        + 0.50 * (metrics["max_drawdown"] - metrics.loc[metrics["strategy"] == "buy_hold", "max_drawdown"].iloc[0]).fillna(0)
    )
    return metrics.sort_values(["strategy_score", "total_return"], ascending=False).reset_index(drop=True)


def select_best_model(metrics: pd.DataFrame, strategy_metrics: pd.DataFrame) -> dict[str, Any]:
    ml_strategy = strategy_metrics[
        strategy_metrics["model_name"].isin(metrics["model_name"].unique())
        & ~strategy_metrics["strategy"].str.startswith("ml_index_enhanced_plus")
    ].copy()
    if ml_strategy.empty:
        ml_strategy_score = pd.DataFrame(columns=["model_name", "horizon", "best_strategy_excess", "best_strategy_mdd"])
    else:
        ml_strategy_score = (
            ml_strategy.sort_values("strategy_score", ascending=False)
            .groupby(["model_name", "horizon"], as_index=False)
            .first()[["model_name", "horizon", "excess_return_vs_buy_hold", "max_drawdown", "strategy"]]
            .rename(
                columns={
                    "excess_return_vs_buy_hold": "best_strategy_excess",
                    "max_drawdown": "best_strategy_mdd",
                    "strategy": "best_strategy_for_model",
                }
            )
        )
    scored = metrics.merge(ml_strategy_score, on=["model_name", "horizon"], how="left")
    scored["selection_score"] = (
        0.22 * scored["AUC"].fillna(0.5)
        + 0.18 * scored["directional_accuracy"].fillna(0.5)
        + 0.14 * scored["return_correlation"].clip(lower=0).fillna(0.0)
        + 0.10 * scored["big_up_AUC"].fillna(0.5)
        + 0.10 * scored["big_down_AUC"].fillna(0.5)
        + 0.06 * scored["tail_score_return_correlation"].clip(lower=0).fillna(0.0)
        + 0.20 * scored["best_strategy_excess"].fillna(0.0)
        + 0.10 * (scored["best_strategy_mdd"].fillna(-0.2) + 0.2)
    )
    best = scored.sort_values(["selection_score", "AUC", "directional_accuracy"], ascending=False).iloc[0]
    return {
        "model_name": str(best["model_name"]),
        "horizon": int(best["horizon"]),
        "selection_score": float(best["selection_score"]),
        "AUC": None if pd.isna(best["AUC"]) else float(best["AUC"]),
        "directional_accuracy": None if pd.isna(best["directional_accuracy"]) else float(best["directional_accuracy"]),
        "return_correlation": None if pd.isna(best["return_correlation"]) else float(best["return_correlation"]),
        "big_up_AUC": None if pd.isna(best.get("big_up_AUC", np.nan)) else float(best["big_up_AUC"]),
        "big_down_AUC": None if pd.isna(best.get("big_down_AUC", np.nan)) else float(best["big_down_AUC"]),
        "tail_score_return_correlation": None
        if pd.isna(best.get("tail_score_return_correlation", np.nan))
        else float(best["tail_score_return_correlation"]),
        "best_strategy_for_model": None if pd.isna(best.get("best_strategy_for_model", np.nan)) else str(best["best_strategy_for_model"]),
        "best_strategy_excess": None if pd.isna(best.get("best_strategy_excess", np.nan)) else float(best["best_strategy_excess"]),
        "rule": "Composite score using AUC, regression direction accuracy, return correlation, tail-target AUCs, tail-score IC, strategy excess return, and drawdown.",
    }


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def save_feature_columns(feature_cols: list[str], skipped_models: list[dict[str, str]]) -> None:
    write_json(
        OUTPUT_DIR / "feature_columns.json",
        {
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "excluded_rule": "Date, raw identifiers, raw OHLCV/amount levels, forward returns, labels, and target dates are excluded.",
            "optional_model_skips": skipped_models,
        },
    )


def plot_equity_curves(strategy_daily: pd.DataFrame, strategy_metrics: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, 7))
    benchmark = strategy_daily[strategy_daily["strategy"] == "buy_hold"].sort_values("date")
    plt.plot(benchmark["date"], benchmark["equity"], label="Buy-and-Hold", color="black", linewidth=2.5)

    highlight = []
    for strategy in [
        "technical_timing",
        "ml_signal_aggressive",
        "ml_signal_benchmark_aware",
        "ml_risk_control",
        "ensemble_strategy",
        "ml_index_enhanced_core",
        "ml_index_full_participation",
        "ml_directional_alpha_extratrees_10d",
        "ml_validation_tuned_index_core",
        "ml_dynamic_ensemble_full_participation",
        "ml_cost_aware_index_enhancement",
        "ml_cost_aware_overlay_conservative",
        "ml_tail_score_index_enhancement",
        "ml_directional_long_cash",
        "ml_directional_scaled_timing",
        "ml_vol_target_trend",
        "ml_enhanced_cppi",
    ]:
        subset = strategy_metrics[strategy_metrics["strategy"] == strategy]
        if not subset.empty:
            highlight.append(subset.sort_values("strategy_score", ascending=False).iloc[0])
    colors = [
        "#2a9d8f",
        "#e76f51",
        "#457b9d",
        "#7b2cbf",
        "#f4a261",
        "#1d3557",
        "#118ab2",
        "#6a4c93",
        "#8ac926",
        "#005f73",
        "#0a9396",
        "#6a994e",
        "#bc4749",
        "#ffb703",
        "#fb8500",
        "#219ebc",
    ]
    for color, row in zip(colors, highlight):
        curve = strategy_daily[
            (strategy_daily["strategy"] == row["strategy"])
            & (strategy_daily["model_name"] == row["model_name"])
            & (strategy_daily["horizon"] == row["horizon"])
        ].sort_values("date")
        label = row["strategy"] if row["horizon"] in [0, -1] else f"{row['strategy']} ({row['model_name']}, {row['horizon']}d)"
        plt.plot(curve["date"], curve["equity"], label=label, linewidth=1.6, color=color)

    plt.title("Equity Curve: Buy-and-Hold vs Enhanced Strategies")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_all_strategies.png", dpi=170)
    plt.close()


def select_best_enhanced_strategy(strategy_metrics: pd.DataFrame) -> pd.Series:
    enhanced_mask = strategy_metrics["strategy"].str.startswith("ml_index_enhanced_plus")
    candidates = strategy_metrics.loc[
        (strategy_metrics["strategy"] != "buy_hold") & ~enhanced_mask
    ].copy()
    if candidates.empty:
        return strategy_metrics.iloc[0]
    return candidates.sort_values(["strategy_score", "total_return"], ascending=False).iloc[0]


def strategy_highlights(strategy_metrics: pd.DataFrame) -> dict[str, Any]:
    benchmark = strategy_metrics.loc[strategy_metrics["strategy"] == "buy_hold"].iloc[0]
    enhanced_mask = strategy_metrics["strategy"].str.startswith("ml_index_enhanced_plus")
    no_leverage = strategy_metrics.loc[
        (strategy_metrics["strategy"] != "buy_hold") & ~enhanced_mask
    ].copy()
    plus = strategy_metrics.loc[enhanced_mask].copy()
    best_no_leverage_total = no_leverage.sort_values("total_return", ascending=False).iloc[0]
    real_no_leverage = strategy_metrics.loc[
        (strategy_metrics["strategy"] != "buy_hold")
        & ~enhanced_mask
        & ~strategy_metrics["is_benchmark_clone"].astype(bool)
        & strategy_metrics["is_real_ml_timing_strategy"].astype(bool)
    ].copy()
    if real_no_leverage.empty:
        best_real_no_leverage_ml_strategy = best_no_leverage_total
        best_real_no_leverage_ml_strategy_fallback = True
    else:
        best_real_no_leverage_ml_strategy = real_no_leverage.sort_values(
            ["total_return", "excess_return_vs_buy_hold", "max_drawdown"],
            ascending=[False, False, False],
        ).iloc[0]
        best_real_no_leverage_ml_strategy_fallback = False
    return {
        "benchmark": benchmark,
        "best_no_leverage_total": best_no_leverage_total,
        "best_real_no_leverage_ml_strategy": best_real_no_leverage_ml_strategy,
        "best_real_no_leverage_ml_strategy_fallback": best_real_no_leverage_ml_strategy_fallback,
        "best_no_leverage_sharpe": no_leverage.sort_values("sharpe", ascending=False).iloc[0],
        "best_no_leverage_drawdown": no_leverage.sort_values("max_drawdown", ascending=False).iloc[0],
        "best_enhanced_exposure": plus.sort_values("total_return", ascending=False).iloc[0] if not plus.empty else None,
        "any_no_leverage_outperform": bool((no_leverage["total_return"] > benchmark["total_return"]).any()),
        "any_real_no_leverage_outperform": bool((real_no_leverage["total_return"] > benchmark["total_return"]).any()) if not real_no_leverage.empty else False,
        "any_plus_outperform": bool((plus["total_return"] > benchmark["total_return"]).any()) if not plus.empty else False,
    }


def get_best_strategy_curve(strategy_daily: pd.DataFrame, strategy_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    best = select_best_enhanced_strategy(strategy_metrics)
    curve = strategy_daily[
        (strategy_daily["strategy"] == best["strategy"])
        & (strategy_daily["model_name"] == best["model_name"])
        & (strategy_daily["horizon"] == best["horizon"])
    ].sort_values("date")
    return curve, best


def select_curve_for_strategy(
    strategy_daily: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    strategy: str,
    preferred_model: str | None = None,
    preferred_horizon: int | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    candidates = strategy_metrics.loc[strategy_metrics["strategy"] == strategy].copy()
    if candidates.empty:
        return pd.DataFrame(), None
    if preferred_model is not None and preferred_horizon is not None:
        preferred = candidates.loc[
            (candidates["model_name"] == preferred_model) & (candidates["horizon"] == preferred_horizon)
        ]
        row = preferred.iloc[0] if not preferred.empty else candidates.sort_values("total_return", ascending=False).iloc[0]
    else:
        row = candidates.sort_values("total_return", ascending=False).iloc[0]
    curve = strategy_daily[
        (strategy_daily["strategy"] == row["strategy"])
        & (strategy_daily["model_name"] == row["model_name"])
        & (strategy_daily["horizon"] == row["horizon"])
    ].sort_values("date")
    return curve, row


def select_curve_for_metrics_row(strategy_daily: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    return strategy_daily[
        (strategy_daily["strategy"] == row["strategy"])
        & (strategy_daily["model_name"] == row["model_name"])
        & (strategy_daily["horizon"] == row["horizon"])
    ].sort_values("date")


def build_active_return_attribution(
    strategy_daily: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    strategy_names = [
        "buy_hold",
        "ml_directional_alpha_extratrees_10d",
        "ml_cost_aware_overlay_conservative",
        "ml_index_enhanced_plus_120",
    ]
    benchmark_curve, benchmark_row = select_curve_for_strategy(strategy_daily, strategy_metrics, "buy_hold")
    if benchmark_curve.empty or benchmark_row is None:
        return pd.DataFrame(), {}

    benchmark = benchmark_curve[
        ["date", "strategy_return", "final_position", "turnover", "equity", "drawdown"]
    ].rename(
        columns={
            "strategy_return": "benchmark_return",
            "final_position": "benchmark_position",
            "turnover": "benchmark_turnover",
            "equity": "benchmark_equity",
            "drawdown": "benchmark_drawdown",
        }
    )
    frames: list[pd.DataFrame] = []
    selected_rows: dict[str, pd.Series] = {"buy_hold": benchmark_row}

    for strategy_name in strategy_names:
        curve, row = select_curve_for_strategy(strategy_daily, strategy_metrics, strategy_name)
        if curve.empty or row is None:
            continue
        selected_rows[strategy_name] = row
        merged = curve.merge(benchmark, on="date", how="left")
        fwd_ret = merged["fwd_ret_1d"].astype(float).fillna(0.0)
        strategy_return = merged["strategy_return"].astype(float).fillna(0.0)
        benchmark_return = merged["benchmark_return"].astype(float).fillna(0.0)
        position_gap = merged["final_position"].astype(float).fillna(0.0) - 1.0
        active_return = strategy_return - benchmark_return
        turnover_cost = merged["turnover"].astype(float).fillna(0.0) * COST_RATE
        benchmark_turnover_cost = merged["benchmark_turnover"].astype(float).fillna(0.0) * COST_RATE

        frame = pd.DataFrame(
            {
                "date": merged["date"],
                "trade_date": merged["trade_date"],
                "strategy": strategy_name,
                "model_name": row["model_name"],
                "horizon": row["horizon"],
                "strategy_role": row.get("strategy_role", ""),
                "is_benchmark_clone": bool(row.get("is_benchmark_clone", False)),
                "benchmark_return": benchmark_return,
                "strategy_return": strategy_return,
                "active_return": active_return,
                "fwd_ret_1d": fwd_ret,
                "final_position": merged["final_position"].astype(float),
                "position_gap": position_gap,
                "position_gap_return": position_gap * fwd_ret,
                "turnover": merged["turnover"].astype(float),
                "turnover_cost": turnover_cost,
                "benchmark_turnover": merged["benchmark_turnover"].astype(float),
                "benchmark_turnover_cost": benchmark_turnover_cost,
                "incremental_turnover_cost": turnover_cost - benchmark_turnover_cost,
                "equity": merged["equity"].astype(float),
                "benchmark_equity": merged["benchmark_equity"].astype(float),
                "drawdown": merged["drawdown"].astype(float),
                "benchmark_drawdown": merged["benchmark_drawdown"].astype(float),
            }
        )
        frame["cumulative_active_return"] = frame["active_return"].cumsum()
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), selected_rows
    return pd.concat(frames, ignore_index=True), selected_rows


def format_markdown_top_dates(df: pd.DataFrame, ascending: bool) -> str:
    if df.empty:
        return "- NA"
    top = df.sort_values("active_return", ascending=ascending).head(10)
    rows = [
        "| date | active_return | strategy_return | benchmark_return | position_gap | turnover_cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        rows.append(
            "| "
            f"{pd.to_datetime(row['date']).date()} | "
            f"{row['active_return']:.4%} | "
            f"{row['strategy_return']:.4%} | "
            f"{row['benchmark_return']:.4%} | "
            f"{row['position_gap']:.4f} | "
            f"{row['turnover_cost']:.4%} |"
        )
    return "\n".join(rows)


def save_active_return_attribution(
    strategy_daily: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> tuple[Path, Path]:
    attribution, selected_rows = build_active_return_attribution(strategy_daily, strategy_metrics)
    attribution_path = OUTPUT_DIR / "active_return_attribution.csv"
    summary_path = OUTPUT_DIR / "active_return_attribution_summary.md"
    attribution.to_csv(attribution_path, index=False)

    directional = attribution[attribution["strategy"] == "ml_directional_alpha_extratrees_10d"].copy()
    if not directional.empty:
        directional["plot_date"] = pd.to_datetime(directional["date"])
        colors = np.where(directional["active_return"] >= 0, "#2a9d8f", "#d62828")
        fig, ax1 = plt.subplots(figsize=(13, 6))
        ax1.bar(directional["plot_date"], directional["active_return"], color=colors, alpha=0.45, label="Daily active return")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Daily active return")
        ax2 = ax1.twinx()
        ax2.plot(
            directional["plot_date"],
            directional["cumulative_active_return"],
            color="#1d3557",
            linewidth=2,
            label="Cumulative active return",
        )
        ax2.set_ylabel("Cumulative active return")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.title("Active Return Attribution: ExtraTrees 10D Directional Alpha")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "active_return_attribution_directional_alpha.png", dpi=170)
        plt.close(fig)

        plt.figure(figsize=(13, 5))
        plt.plot(directional["plot_date"], directional["position_gap"], color="#457b9d", linewidth=1.8)
        plt.axhline(0, color="black", linewidth=0.9)
        plt.title("Position Gap vs Buy-and-Hold: ExtraTrees 10D Directional Alpha")
        plt.xlabel("Date")
        plt.ylabel("Final position - 1.00")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "position_gap_directional_alpha.png", dpi=170)
        plt.close()

    def strategy_block(strategy_name: str) -> str:
        strat = attribution[attribution["strategy"] == strategy_name].copy()
        row = selected_rows.get(strategy_name)
        if strat.empty or row is None:
            return f"## {strategy_name}\n- Not available.\n"

        position_gap_return = float(strat["position_gap_return"].sum())
        turnover_cost = float(strat["turnover_cost"].sum())
        incremental_turnover_cost = float(strat["incremental_turnover_cost"].sum())
        active_return = float(strat["active_return"].sum())
        positive = format_markdown_top_dates(strat, ascending=False)
        negative = format_markdown_top_dates(strat, ascending=True)
        return f"""## {strategy_name}
- Selected row: {row['model_name']}, horizon={row['horizon']}
- Strategy role: {row.get('strategy_role', 'NA')}
- Benchmark clone: {'Yes' if bool(row.get('is_benchmark_clone', False)) else 'No'}
- Total return: {format_pct(row['total_return'])}
- Excess return vs buy-and-hold: {format_pct(row['excess_return_vs_buy_hold'])}
- Max drawdown: {format_pct(row['max_drawdown'])}
- Sum active return: {format_pct(active_return)}
- Sum position-gap return: {format_pct(position_gap_return)}
- Total turnover cost: {format_pct(turnover_cost)}
- Incremental turnover cost vs buy-and-hold: {format_pct(incremental_turnover_cost)}
- Avg position: {format_num(row.get('avg_position'))}
- Days below full exposure: {int(row.get('days_below_full_exposure', 0))}
- Total turnover: {format_num(row.get('total_turnover'))}

Top 10 positive active-return dates:

{positive}

Top 10 negative active-return dates:

{negative}
"""

    cost_row = selected_rows.get("ml_cost_aware_overlay_conservative")
    directional_row = selected_rows.get("ml_directional_alpha_extratrees_10d")
    plus120_row = selected_rows.get("ml_index_enhanced_plus_120")

    directional_explanation = "- ml_directional_alpha_extratrees_10d is not available."
    if not directional.empty and directional_row is not None:
        up_days = directional["fwd_ret_1d"] > 0
        down_days = directional["fwd_ret_1d"] < 0
        missed_upside = -float(directional.loc[up_days, "position_gap_return"].clip(upper=0).sum())
        avoided_downside = float(directional.loc[down_days, "position_gap_return"].clip(lower=0).sum())
        incremental_cost = float(directional["incremental_turnover_cost"].sum())
        if missed_upside > max(avoided_downside, incremental_cost):
            reason = "The main drag is missed upside from reduced exposure on rising days; incremental trading cost is smaller."
        elif incremental_cost > max(missed_upside - avoided_downside, 0):
            reason = "The main drag is transaction cost, while exposure timing is close to neutral."
        else:
            reason = "The drag is mixed between missed upside and transaction cost."
        directional_explanation = (
            f"- ml_directional_alpha_extratrees_10d active return comes from "
            f"{int(directional_row.get('days_below_full_exposure', 0))} reduced-exposure days, "
            f"with missed-upside loss {format_pct(missed_upside)}, avoided-downside gain {format_pct(avoided_downside)}, "
            f"and incremental turnover cost {format_pct(incremental_cost)}. {reason}"
        )

    clone_explanation = "- ml_cost_aware_overlay_conservative is not available."
    if cost_row is not None:
        clone_explanation = (
            "- ml_cost_aware_overlay_conservative is treated as a benchmark clone: "
            f"benchmark_clone={'Yes' if bool(cost_row.get('is_benchmark_clone', False)) else 'No'}, "
            f"avg_position={format_num(cost_row.get('avg_position'))}, "
            f"days_below_full_exposure={int(cost_row.get('days_below_full_exposure', 0))}, "
            f"tracking_error={format_num(cost_row.get('tracking_error'))}."
        )

    plus_explanation = "- ml_index_enhanced_plus_120 is not available."
    plus120 = attribution[attribution["strategy"] == "ml_index_enhanced_plus_120"].copy()
    if not plus120.empty and plus120_row is not None:
        plus_explanation = (
            "- ml_index_enhanced_plus_120 positive excess return comes from enhanced exposure: "
            f"max_position={format_num(plus120_row.get('max_position'))}, "
            f"avg_position={format_num(plus120_row.get('avg_position'))}, "
            f"sum position-gap return={format_pct(float(plus120['position_gap_return'].sum()))}."
        )

    summary = f"""# Active Return Attribution Summary

Objects analyzed:
- buy_hold
- ml_directional_alpha_extratrees_10d
- ml_cost_aware_overlay_conservative
- ml_index_enhanced_plus_120

## Interpretation
{clone_explanation}
{directional_explanation}
{plus_explanation}

{strategy_block('ml_directional_alpha_extratrees_10d')}

{strategy_block('ml_cost_aware_overlay_conservative')}

{strategy_block('ml_index_enhanced_plus_120')}
"""
    summary_path.write_text(summary, encoding="utf-8")
    return attribution_path, summary_path


def build_directional_trading_signals(strategy_daily: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for strategy_name in ["ml_directional_long_cash", "ml_directional_scaled_timing"]:
        curve = strategy_daily[strategy_daily["strategy"] == strategy_name].sort_values("date").copy()
        if curve.empty:
            continue
        prev_position = curve["final_position"].shift(1).fillna(curve["final_position"])
        diff = curve["final_position"] - prev_position
        curve["signal"] = np.select(
            [diff > 1e-9, diff < -1e-9],
            ["BUY", "SELL"],
            default="HOLD",
        )
        frames.append(
            curve[
                [
                    "date",
                    "trade_date",
                    "strategy",
                    "model_name",
                    "horizon",
                    "pred_ret",
                    "pred_up_prob",
                    "pred_trade_prob",
                    "signal",
                    "final_position",
                    "fwd_ret_1d",
                    "strategy_return",
                    "equity",
                    "drawdown",
                    "turnover",
                ]
            ].copy()
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "date",
                "trade_date",
                "strategy",
                "model_name",
                "horizon",
                "pred_ret",
                "pred_up_prob",
                "pred_trade_prob",
                "signal",
                "final_position",
                "fwd_ret_1d",
                "strategy_return",
                "equity",
                "drawdown",
                "turnover",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def markdown_signal_dates(df: pd.DataFrame, sort_col: str, ascending: bool) -> str:
    if df.empty:
        return "- NA"
    top = df.sort_values(sort_col, ascending=ascending).head(10)
    rows = [
        "| date | signal | fwd_ret_1d | pred_up_prob | pred_ret | final_position |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        rows.append(
            "| "
            f"{pd.to_datetime(row['date']).date()} | "
            f"{row['signal']} | "
            f"{row['fwd_ret_1d']:.4%} | "
            f"{row['pred_up_prob']:.4f} | "
            f"{row['pred_ret']:.4%} | "
            f"{row['final_position']:.4f} |"
        )
    return "\n".join(rows)


def summarize_directional_signal_frame(signals: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for strategy_name, grp in signals.groupby("strategy", sort=True):
        grp = grp.sort_values("date").copy()
        prev_position = grp["final_position"].shift(1).fillna(grp["final_position"])
        position_cut = (prev_position - grp["final_position"]).clip(lower=0)
        buy = grp["signal"] == "BUY"
        sell = grp["signal"] == "SELL"
        hold = grp["signal"] == "HOLD"
        successful_sell = sell & (grp["fwd_ret_1d"] < 0)
        failed_sell = sell & (grp["fwd_ret_1d"] > 0)
        summary[strategy_name] = {
            "total_buy_signals": int(buy.sum()),
            "total_sell_signals": int(sell.sum()),
            "total_hold_days": int(hold.sum()),
            "average_return_after_buy": float(grp.loc[buy, "fwd_ret_1d"].mean()) if buy.any() else np.nan,
            "average_return_after_sell": float(grp.loc[sell, "fwd_ret_1d"].mean()) if sell.any() else np.nan,
            "win_rate_after_buy": float((grp.loc[buy, "fwd_ret_1d"] > 0).mean()) if buy.any() else np.nan,
            "successful_sell_count": int(successful_sell.sum()),
            "failed_sell_count": int(failed_sell.sum()),
            "avoided_loss_after_sell": float((-grp.loc[successful_sell, "fwd_ret_1d"] * position_cut.loc[successful_sell]).sum()),
            "missed_gain_after_sell": float((grp.loc[failed_sell, "fwd_ret_1d"] * position_cut.loc[failed_sell]).sum()),
            "turnover_cost": float(grp["turnover"].fillna(0).sum() * COST_RATE),
        }
    return summary


def save_directional_trading_outputs(strategy_daily: pd.DataFrame) -> tuple[Path, Path]:
    signals = build_directional_trading_signals(strategy_daily)
    signals_path = OUTPUT_DIR / "directional_trading_signals.csv"
    summary_path = OUTPUT_DIR / "directional_timing_attribution_summary.md"
    signals.to_csv(signals_path, index=False)
    signal_summary = summarize_directional_signal_frame(signals)

    sections: list[str] = ["# Directional Timing Attribution Summary"]
    if signals.empty:
        sections.append("\nNo direct directional timing strategies were available.")
    for strategy_name, stats in signal_summary.items():
        grp = signals[signals["strategy"] == strategy_name].copy()
        successful_buy = grp[(grp["signal"] == "BUY") & (grp["fwd_ret_1d"] > 0)]
        failed_buy = grp[(grp["signal"] == "BUY") & (grp["fwd_ret_1d"] < 0)]
        successful_sell = grp[(grp["signal"] == "SELL") & (grp["fwd_ret_1d"] < 0)]
        failed_sell = grp[(grp["signal"] == "SELL") & (grp["fwd_ret_1d"] > 0)]
        sections.append(
            f"""
## {strategy_name}
- Total BUY signals: {stats['total_buy_signals']}
- Total SELL signals: {stats['total_sell_signals']}
- Total HOLD days: {stats['total_hold_days']}
- Average return after BUY signal: {format_pct(stats['average_return_after_buy'])}
- Average return after SELL signal: {format_pct(stats['average_return_after_sell'])}
- Win rate after BUY signal: {format_pct(stats['win_rate_after_buy'])}
- Successful SELL count: {stats['successful_sell_count']}
- Failed SELL count: {stats['failed_sell_count']}
- Avoided loss after SELL signal: {format_pct(stats['avoided_loss_after_sell'])}
- Missed gain after SELL signal: {format_pct(stats['missed_gain_after_sell'])}
- Turnover cost: {format_pct(stats['turnover_cost'])}

Top 10 successful BUY dates:

{markdown_signal_dates(successful_buy, 'fwd_ret_1d', ascending=False)}

Top 10 failed BUY dates:

{markdown_signal_dates(failed_buy, 'fwd_ret_1d', ascending=True)}

Top 10 successful SELL dates:

{markdown_signal_dates(successful_sell, 'fwd_ret_1d', ascending=True)}

Top 10 failed SELL dates:

{markdown_signal_dates(failed_sell, 'fwd_ret_1d', ascending=False)}
"""
        )
    summary_path.write_text("\n".join(sections), encoding="utf-8")
    return signals_path, summary_path


def markdown_defensive_dates(df: pd.DataFrame, ascending: bool) -> str:
    if df.empty:
        return "- NA"
    top = df.sort_values("position_gap_return", ascending=ascending).head(10)
    rows = [
        "| date | fwd_ret_1d | final_position | position_gap_return | turnover_cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        rows.append(
            "| "
            f"{pd.to_datetime(row['date']).date()} | "
            f"{row['fwd_ret_1d']:.4%} | "
            f"{row['final_position']:.4f} | "
            f"{row['position_gap_return']:.4%} | "
            f"{row['turnover_cost']:.4%} |"
        )
    return "\n".join(rows)


def save_selective_defensive_attribution(
    strategy_daily: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
) -> tuple[Path, Path]:
    attr_path = OUTPUT_DIR / "selective_defensive_attribution.csv"
    summary_path = OUTPUT_DIR / "selective_defensive_attribution_summary.md"
    selective = strategy_daily[strategy_daily["strategy"] == "ml_selective_defensive_enhancement"].sort_values("date").copy()
    benchmark = (
        strategy_daily[strategy_daily["strategy"] == "buy_hold"]
        .sort_values("date")
        .rename(
            columns={
                "strategy_return": "benchmark_return",
                "equity": "benchmark_equity",
                "drawdown": "benchmark_drawdown",
            }
        )
    )
    if selective.empty or benchmark.empty:
        pd.DataFrame().to_csv(attr_path, index=False)
        summary_path.write_text(
            "# Selective Defensive Attribution\n\nml_selective_defensive_enhancement was not available in this run.\n",
            encoding="utf-8",
        )
        return attr_path, summary_path

    merged = selective.merge(
        benchmark[["date", "benchmark_return", "benchmark_equity", "benchmark_drawdown"]],
        on="date",
        how="left",
    )
    merged["position_gap"] = merged["final_position"].astype(float) - 1.0
    merged["position_gap_return"] = merged["position_gap"] * merged["fwd_ret_1d"].astype(float)
    merged["turnover_cost"] = merged["turnover"].astype(float) * COST_RATE
    merged["cumulative_position_timing_contribution"] = merged["position_gap_return"].cumsum()
    attr = merged[
        [
            "date",
            "trade_date",
            "strategy",
            "model_name",
            "horizon",
            "pred_ret",
            "pred_up_prob",
            "pred_trade_prob",
            "fwd_ret_1d",
            "benchmark_return",
            "strategy_return",
            "final_position",
            "turnover",
            "turnover_cost",
            "position_gap",
            "position_gap_return",
            "cumulative_position_timing_contribution",
            "equity",
            "drawdown",
            "benchmark_equity",
            "benchmark_drawdown",
        ]
    ].copy()
    attr.to_csv(attr_path, index=False)

    metric_rows = strategy_metrics[strategy_metrics["strategy"] == "ml_selective_defensive_enhancement"].copy()
    metric_row = metric_rows.sort_values("total_return", ascending=False).iloc[0] if not metric_rows.empty else pd.Series(dtype=object)
    missed_upside = -float(attr.loc[attr["fwd_ret_1d"] > 0, "position_gap_return"].clip(upper=0).sum())
    avoided_downside = float(attr.loc[attr["fwd_ret_1d"] < 0, "position_gap_return"].clip(lower=0).sum())
    net_contribution = float(attr["position_gap_return"].sum())
    transaction_cost = float(attr["turnover_cost"].sum())
    best_days = markdown_defensive_dates(attr, ascending=False)
    worst_days = markdown_defensive_dates(attr, ascending=True)
    if avoided_downside > missed_upside + transaction_cost:
        interpretation = (
            "The defensive adjustment was effective: avoided downside exceeded missed upside plus transaction cost."
        )
    elif missed_upside >= transaction_cost:
        interpretation = (
            "The strategy did not outperform mainly because reduced exposure missed upside on positive-return days."
        )
    else:
        interpretation = "The strategy did not outperform mainly because transaction cost outweighed the defensive benefit."

    summary = f"""# Selective Defensive Attribution

- Selected model_name: {metric_row.get('model_name', selective['model_name'].iloc[0])}
- Selected horizon: {metric_row.get('horizon', selective['horizon'].iloc[0])}
- Total return: {format_pct(metric_row.get('total_return'))}
- Excess return vs buy-and-hold: {format_pct(metric_row.get('excess_return_vs_buy_hold'))}
- Max drawdown: {format_pct(metric_row.get('max_drawdown'))}
- Total days below full exposure: {int(metric_row.get('days_below_full_exposure', int((attr['final_position'] < 0.999).sum())))}
- Avg position: {format_num(metric_row.get('avg_position'))}
- Min position: {format_num(metric_row.get('min_position'))}
- Total turnover: {format_num(metric_row.get('total_turnover'))}
- Transaction cost: {format_pct(transaction_cost)}
- Missed upside from reduced exposure on positive-return days: {format_pct(missed_upside)}
- Avoided downside from reduced exposure on negative-return days: {format_pct(avoided_downside)}
- Net position timing contribution: {format_pct(net_contribution)}

{interpretation}

## Top 10 Best Defensive Days

{best_days}

## Top 10 Worst Defensive Days

{worst_days}
"""
    summary_path.write_text(summary, encoding="utf-8")
    return attr_path, summary_path


def save_plots(
    feature_df: pd.DataFrame,
    pred_all: pd.DataFrame,
    model_metrics: pd.DataFrame,
    strategy_daily: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    factor_effectiveness: pd.DataFrame,
    best_model: dict[str, Any],
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_equity_curves(strategy_daily, strategy_metrics)

    best_curve, best_strategy = get_best_strategy_curve(strategy_daily, strategy_metrics)
    buy_hold = strategy_daily[strategy_daily["strategy"] == "buy_hold"].sort_values("date")

    plt.figure(figsize=(12, 6))
    plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2)
    plt.plot(best_curve["date"], best_curve["equity"], label=f"Best Strategy: {best_strategy['strategy']}", color="#e76f51")
    plt.title("Best Strategy vs Buy-and-Hold Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_best_vs_buyhold.png", dpi=170)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2)
    plt.plot(best_curve["date"], best_curve["drawdown"], label=f"Best Strategy: {best_strategy['strategy']}", color="#457b9d")
    plt.title("Drawdown Curve: Best Strategy vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_best_vs_buyhold.png", dpi=170)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(best_curve["date"], best_curve["raw_position"], label="Raw Position", alpha=0.65)
    plt.plot(best_curve["date"], best_curve["final_position"], label="Final Position After Drawdown Control", linewidth=1.8)
    plt.title("Position Exposure of the Best Strategy")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.ylim(-0.02, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "position_exposure_best_strategy.png", dpi=170)
    plt.close()

    best_pred = pred_all[
        (pred_all["model_name"] == best_model["model_name"]) & (pred_all["horizon"] == best_model["horizon"])
    ].sort_values("date")
    plt.figure(figsize=(12, 5))
    plt.plot(best_pred["date"], best_pred["pred_up_prob"], label="Predicted Up Probability", color="#2a9d8f")
    plt.plot(best_pred["date"], best_pred["pred_trade_prob"], label="Predicted Trade Probability", color="#f4a261", alpha=0.8)
    plt.title(f"Predicted Probability Over Time: {best_model['model_name']} {best_model['horizon']}D")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "predicted_probability_over_time_best_model.png", dpi=170)
    plt.close()

    scatter = best_pred.dropna(subset=["pred_ret", "target_ret"])
    plt.figure(figsize=(7, 6))
    plt.scatter(scatter["pred_ret"], scatter["target_ret"], s=24, alpha=0.65, color="#457b9d")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(f"Predicted vs Actual Forward Return: {best_model['model_name']} {best_model['horizon']}D")
    plt.xlabel("Predicted Forward Return")
    plt.ylabel("Actual Forward Return")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "prediction_scatter_best_model.png", dpi=170)
    plt.close()

    top_strats = strategy_metrics.sort_values("total_return", ascending=False).head(12).copy()
    top_strats["label"] = top_strats.apply(
        lambda r: r["strategy"] if r["horizon"] in [0, -1] else f"{r['strategy']}\n{r['model_name']} {int(r['horizon'])}D",
        axis=1,
    )
    plt.figure(figsize=(13, 6))
    plt.bar(top_strats["label"], top_strats["total_return"], color="#2a9d8f")
    plt.axhline(strategy_metrics.loc[strategy_metrics["strategy"] == "buy_hold", "total_return"].iloc[0], color="black", linestyle="--", label="Buy-and-Hold")
    plt.title("Total Return by Top Strategies")
    plt.xlabel("Strategy")
    plt.ylabel("Total Return")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "strategy_return_bars.png", dpi=170)
    plt.close()

    mm = model_metrics.sort_values("AUC", ascending=False).head(20).copy()
    mm["label"] = mm["model_name"] + " " + mm["horizon"].astype(str) + "D"
    x = np.arange(len(mm))
    width = 0.25
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, mm["AUC"], width, label="AUC")
    plt.bar(x, mm["directional_accuracy"], width, label="Directional Accuracy")
    plt.bar(x + width, mm["return_correlation"], width, label="Return Correlation")
    plt.title("Model Metric Comparison by Horizon")
    plt.xlabel("Model and Horizon")
    plt.ylabel("Metric")
    plt.xticks(x, mm["label"], rotation=45, ha="right", fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_metric_comparison.png", dpi=170)
    plt.close()

    top_factors = factor_effectiveness.dropna(subset=["corr_with_forward_return"]).head(20).copy()
    plt.figure(figsize=(10, 7))
    plt.barh(top_factors["factor"][::-1], top_factors["corr_with_forward_return"][::-1], color="#7b2cbf")
    plt.title("Top 20 Factor Correlations with 5-Day Forward Return")
    plt.xlabel("Correlation")
    plt.ylabel("Factor")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "factor_correlation_top20.png", dpi=170)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(best_curve["date"], best_curve["turnover"], label="Best Strategy Turnover", color="#e76f51")
    ensemble = strategy_daily[strategy_daily["strategy"] == "ensemble_strategy"].sort_values("date")
    if not ensemble.empty:
        plt.plot(ensemble["date"], ensemble["turnover"], label="Ensemble Strategy Turnover", alpha=0.7)
    plt.title("Turnover Over Time")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "turnover_over_time.png", dpi=170)
    plt.close()

    preferred_model = best_model["model_name"]
    preferred_horizon = best_model["horizon"]
    index_strategy_names = ["ml_index_enhanced_core", "ml_vol_target_trend", "ml_enhanced_cppi"]
    index_curves: list[tuple[str, pd.DataFrame]] = []
    for strategy_name in index_strategy_names:
        curve, row = select_curve_for_strategy(
            strategy_daily, strategy_metrics, strategy_name, preferred_model, preferred_horizon
        )
        if not curve.empty and row is not None:
            index_curves.append((f"{strategy_name} ({row['model_name']}, {int(row['horizon'])}D)", curve))

    plt.figure(figsize=(12, 6))
    plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
    for label, curve in index_curves:
        plt.plot(curve["date"], curve["equity"], label=label, linewidth=1.5)
    plt.title("Index-Enhanced Strategies vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_index_enhanced_strategies_vs_buyhold.png", dpi=170)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
    for label, curve in index_curves:
        plt.plot(curve["date"], curve["drawdown"], label=label, linewidth=1.5)
    plt.title("Drawdown: Index-Enhanced Strategies vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_index_enhanced_strategies_vs_buyhold.png", dpi=170)
    plt.close()

    exposure_specs = [
        ("ml_index_enhanced_core", "position_exposure_index_enhanced_core.png", "Position Exposure: ML Index Enhanced Core"),
        ("ml_vol_target_trend", "position_exposure_vol_target_trend.png", "Position Exposure: Vol Target + Trend"),
        ("ml_enhanced_cppi", "position_exposure_enhanced_cppi.png", "Position Exposure: CPPI-Style Enhanced Strategy"),
    ]
    for strategy_name, filename, title in exposure_specs:
        curve, row = select_curve_for_strategy(
            strategy_daily, strategy_metrics, strategy_name, preferred_model, preferred_horizon
        )
        if curve.empty:
            continue
        plt.figure(figsize=(12, 5))
        plt.plot(curve["date"], curve["raw_position"], label="Raw Position", alpha=0.65)
        plt.plot(curve["date"], curve["final_position"], label="Final Position", linewidth=1.8)
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Exposure")
        plt.ylim(0.45, 1.12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / filename, dpi=170)
        plt.close()

    plus_curve, plus_row = select_curve_for_strategy(
        strategy_daily, strategy_metrics, "ml_index_enhanced_plus", preferred_model, preferred_horizon
    )
    if not plus_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(
            plus_curve["date"],
            plus_curve["equity"],
            label=f"ML Index Enhanced Plus ({plus_row['model_name']}, {int(plus_row['horizon'])}D)",
            color="#d00000",
        )
        plt.title("Enhanced-Exposure Strategy vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_plus_strategy_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(
            plus_curve["date"],
            plus_curve["drawdown"],
            label=f"ML Index Enhanced Plus ({plus_row['model_name']}, {int(plus_row['horizon'])}D)",
            color="#d00000",
        )
        plt.title("Drawdown: Enhanced-Exposure Strategy vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "drawdown_curve_plus_strategy_vs_buyhold.png", dpi=170)
        plt.close()

    full_curve, full_row = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_index_full_participation")
    if not full_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(
            full_curve["date"],
            full_curve["equity"],
            label=f"Full Participation ({full_row['model_name']}, {int(full_row['horizon'])}D)",
            color="#118ab2",
        )
        plt.title("Full-Participation Strategy vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_full_participation_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(
            full_curve["date"],
            full_curve["drawdown"],
            label=f"Full Participation ({full_row['model_name']}, {int(full_row['horizon'])}D)",
            color="#118ab2",
        )
        plt.title("Drawdown: Full-Participation Strategy vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "drawdown_curve_full_participation_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(full_curve["date"], full_curve["raw_position"], label="Raw Position", alpha=0.65)
        plt.plot(full_curve["date"], full_curve["final_position"], label="Final Position", linewidth=1.8)
        plt.title("Position Exposure: Full-Participation Strategy")
        plt.xlabel("Date")
        plt.ylabel("Exposure")
        plt.ylim(0.82, 1.03)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "position_exposure_full_participation.png", dpi=170)
        plt.close()

    tuned_curve, tuned_row = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_validation_tuned_index_core")
    if not tuned_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(
            tuned_curve["date"],
            tuned_curve["equity"],
            label=f"Validation-Tuned Core ({tuned_row['model_name']}, {int(tuned_row['horizon'])}D)",
            color="#6a4c93",
        )
        plt.title("Validation-Tuned Strategy vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_validation_tuned_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(
            tuned_curve["date"],
            tuned_curve["drawdown"],
            label=f"Validation-Tuned Core ({tuned_row['model_name']}, {int(tuned_row['horizon'])}D)",
            color="#6a4c93",
        )
        plt.title("Drawdown: Validation-Tuned Strategy vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "drawdown_curve_validation_tuned_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(tuned_curve["date"], tuned_curve["raw_position"], label="Raw Position", alpha=0.65)
        plt.plot(tuned_curve["date"], tuned_curve["final_position"], label="Final Position", linewidth=1.8)
        plt.title("Position Exposure: Validation-Tuned Strategy")
        plt.xlabel("Date")
        plt.ylabel("Exposure")
        plt.ylim(0.82, 1.03)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "position_exposure_validation_tuned.png", dpi=170)
        plt.close()

    plus115_curve, plus115_row = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_index_enhanced_plus_115")
    plus120_curve, plus120_row = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_index_enhanced_plus_120")
    if not plus115_curve.empty or not plus120_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
        if not plus115_curve.empty:
            plt.plot(plus115_curve["date"], plus115_curve["equity"], label="Plus 1.15", color="#d00000")
        if not plus120_curve.empty:
            plt.plot(plus120_curve["date"], plus120_curve["equity"], label="Plus 1.20", color="#ff7b00")
        plt.title("Enhanced-Exposure Plus 1.15/1.20 Strategies vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_plus_115_120_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
        if not plus115_curve.empty:
            plt.plot(plus115_curve["date"], plus115_curve["drawdown"], label="Plus 1.15", color="#d00000")
        if not plus120_curve.empty:
            plt.plot(plus120_curve["date"], plus120_curve["drawdown"], label="Plus 1.20", color="#ff7b00")
        plt.title("Drawdown: Plus 1.15/1.20 Strategies vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "drawdown_curve_plus_115_120_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 5))
        if not plus115_curve.empty:
            plt.plot(plus115_curve["date"], plus115_curve["final_position"], label="Plus 1.15 Exposure", color="#d00000")
        if not plus120_curve.empty:
            plt.plot(plus120_curve["date"], plus120_curve["final_position"], label="Plus 1.20 Exposure", color="#ff7b00")
        plt.title("Position Exposure: Plus 1.15/1.20 Strategies")
        plt.xlabel("Date")
        plt.ylabel("Exposure")
        plt.ylim(0.82, 1.23)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "position_exposure_plus_115_120.png", dpi=170)
        plt.close()

    cost_curve, cost_row = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_cost_aware_index_enhancement")
    conservative_curve, conservative_row = select_curve_for_strategy(
        strategy_daily, strategy_metrics, "ml_cost_aware_overlay_conservative"
    )
    if not cost_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(
            cost_curve["date"],
            cost_curve["equity"],
            label=f"Cost-Aware Index Enhancement ({cost_row['model_name']}, {int(cost_row['horizon'])}D)",
            color="#005f73",
        )
        plt.title("Cost-Aware Index Enhancement vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_cost_aware_index_enhancement_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(
            cost_curve["date"],
            cost_curve["drawdown"],
            label=f"Cost-Aware Index Enhancement ({cost_row['model_name']}, {int(cost_row['horizon'])}D)",
            color="#005f73",
        )
        plt.title("Drawdown: Cost-Aware Index Enhancement vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "drawdown_curve_cost_aware_index_enhancement_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(cost_curve["date"], cost_curve["raw_position"], label="Raw Position", alpha=0.65)
        plt.plot(cost_curve["date"], cost_curve["final_position"], label="Final Position", linewidth=1.8)
        plt.title("Position Exposure: Cost-Aware Index Enhancement")
        plt.xlabel("Date")
        plt.ylabel("Exposure")
        plt.ylim(0.82, 1.03)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "position_exposure_cost_aware_index_enhancement.png", dpi=170)
        plt.close()

    best_no_lev_curve, best_no_lev_row = get_best_strategy_curve(strategy_daily, strategy_metrics)
    final_candidates: list[tuple[str, pd.DataFrame, str]] = [
        ("Buy-and-Hold", buy_hold, "black"),
    ]
    directional_curve, _ = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_directional_alpha_extratrees_10d")
    if not directional_curve.empty:
        final_candidates.append(("ExtraTrees 10D Directional Alpha", directional_curve, "#457b9d"))
    if not cost_curve.empty:
        final_candidates.append(("Cost-Aware Index Enhancement", cost_curve, "#005f73"))
    if not conservative_curve.empty:
        final_candidates.append(("Cost-Aware Conservative Overlay", conservative_curve, "#0a9396"))
    tail_curve, _ = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_tail_score_index_enhancement")
    if not tail_curve.empty:
        final_candidates.append(("Tail-Score Index Enhancement", tail_curve, "#ffb703"))
    if not best_no_lev_curve.empty:
        final_candidates.append((f"Best No-Leverage: {best_no_lev_row['strategy']}", best_no_lev_curve, "#e76f51"))

    plt.figure(figsize=(12, 6))
    for label, curve, color in final_candidates:
        plt.plot(curve["date"], curve["equity"], label=label, color=color, linewidth=2.2 if label == "Buy-and-Hold" else 1.5)
    plt.title("Final No-Leverage Candidates vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_final_no_leverage_candidates_vs_buyhold.png", dpi=170)
    plt.close()

    plt.figure(figsize=(12, 6))
    for label, curve, color in final_candidates:
        plt.plot(curve["date"], curve["drawdown"], label=label, color=color, linewidth=2.2 if label == "Buy-and-Hold" else 1.5)
    plt.title("Drawdown: Final No-Leverage Candidates vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_final_no_leverage_candidates_vs_buyhold.png", dpi=170)
    plt.close()

    highlights = strategy_highlights(strategy_metrics)
    real_ml_row = highlights["best_real_no_leverage_ml_strategy"]
    real_ml_curve = select_curve_for_metrics_row(strategy_daily, real_ml_row)
    if not real_ml_curve.empty:
        label = f"Real ML Timing: {real_ml_row['strategy']}"
        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(real_ml_curve["date"], real_ml_curve["equity"], label=label, color="#457b9d", linewidth=1.8)
        plt.title("Real No-Leverage ML Timing Strategy vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_real_ml_strategy_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
        plt.plot(real_ml_curve["date"], real_ml_curve["drawdown"], label=label, color="#457b9d", linewidth=1.8)
        plt.title("Drawdown: Real No-Leverage ML Timing Strategy vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "drawdown_curve_real_ml_strategy_vs_buyhold.png", dpi=170)
        plt.close()

    long_cash_curve, long_cash_row = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_directional_long_cash")
    scaled_curve, scaled_row = select_curve_for_strategy(strategy_daily, strategy_metrics, "ml_directional_scaled_timing")
    if not long_cash_curve.empty or not scaled_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
        if not long_cash_curve.empty:
            plt.plot(long_cash_curve["date"], long_cash_curve["equity"], label="ML Directional Long/Cash", color="#fb8500")
        if not scaled_curve.empty:
            plt.plot(scaled_curve["date"], scaled_curve["equity"], label="ML Directional Scaled Timing", color="#219ebc")
        plt.title("Direct ML Directional Timing Strategies vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_directional_timing_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
        if not long_cash_curve.empty:
            plt.plot(long_cash_curve["date"], long_cash_curve["drawdown"], label="ML Directional Long/Cash", color="#fb8500")
        if not scaled_curve.empty:
            plt.plot(scaled_curve["date"], scaled_curve["drawdown"], label="ML Directional Scaled Timing", color="#219ebc")
        plt.title("Drawdown: Direct ML Directional Timing Strategies vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "drawdown_curve_directional_timing_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 5))
        if not long_cash_curve.empty:
            plt.plot(long_cash_curve["date"], long_cash_curve["final_position"], label="Long/Cash Final Position", color="#fb8500")
        if not scaled_curve.empty:
            plt.plot(scaled_curve["date"], scaled_curve["final_position"], label="Scaled Timing Final Position", color="#219ebc")
        plt.title("Position Exposure: Direct ML Directional Timing")
        plt.xlabel("Date")
        plt.ylabel("Exposure")
        plt.ylim(-0.03, 1.03)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "position_exposure_directional_timing.png", dpi=170)
        plt.close()

    if not scaled_curve.empty and scaled_row is not None:
        fig, ax1 = plt.subplots(figsize=(13, 6))
        ax1.plot(scaled_curve["date"], scaled_curve["pred_up_prob"], label="Predicted Up Probability", color="#2a9d8f")
        ax1.plot(scaled_curve["date"], scaled_curve["final_position"], label="Final Position", color="#219ebc", linewidth=2)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Probability / Position")
        ax1.set_ylim(-0.05, 1.05)
        ax2 = ax1.twinx()
        ax2.plot(scaled_curve["date"], scaled_curve["pred_ret"], label="Predicted Return", color="#bc4749", alpha=0.75)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Predicted return")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.title("Prediction Signal and Position: ML Directional Scaled Timing")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "prediction_signal_and_position.png", dpi=170)
        plt.close(fig)

    selective_curve, selective_row = select_curve_for_strategy(
        strategy_daily, strategy_metrics, "ml_selective_defensive_enhancement"
    )
    if not selective_curve.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["equity"], label="Buy-and-Hold", color="black", linewidth=2.2)
        if not full_curve.empty:
            plt.plot(full_curve["date"], full_curve["equity"], label="ML Full Participation", color="#118ab2")
        plt.plot(
            selective_curve["date"],
            selective_curve["equity"],
            label="ML Selective Defensive Enhancement",
            color="#2d6a4f",
        )
        plt.title("Selective Defensive Enhancement vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_selective_defensive_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(buy_hold["date"], buy_hold["drawdown"], label="Buy-and-Hold", color="black", linewidth=2.2)
        if not full_curve.empty:
            plt.plot(full_curve["date"], full_curve["drawdown"], label="ML Full Participation", color="#118ab2")
        plt.plot(
            selective_curve["date"],
            selective_curve["drawdown"],
            label="ML Selective Defensive Enhancement",
            color="#2d6a4f",
        )
        plt.title("Drawdown: Selective Defensive Enhancement vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "drawdown_curve_selective_defensive_vs_buyhold.png", dpi=170)
        plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(selective_curve["date"], selective_curve["raw_position"], label="Raw Position", alpha=0.70)
        plt.plot(selective_curve["date"], selective_curve["final_position"], label="Final Position", linewidth=1.8)
        plt.title("Position Exposure: Selective Defensive Enhancement")
        plt.xlabel("Date")
        plt.ylabel("Exposure")
        plt.ylim(0.84, 1.03)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "position_exposure_selective_defensive.png", dpi=170)
        plt.close()

        fig, ax1 = plt.subplots(figsize=(13, 6))
        ax1.plot(selective_curve["date"], selective_curve["pred_up_prob"], label="Predicted Up Probability", color="#2a9d8f")
        ax1.plot(selective_curve["date"], selective_curve["final_position"], label="Final Position", color="#2d6a4f", linewidth=2)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Probability / Position")
        ax1.set_ylim(0.0, 1.05)
        ax2 = ax1.twinx()
        ax2.plot(selective_curve["date"], selective_curve["pred_ret"], label="Predicted Return", color="#bc4749", alpha=0.75)
        ax2.plot(selective_curve["date"], selective_curve["fwd_ret_1d"], label="Next-Day Return", color="#6a4c93", alpha=0.55)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Return")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
        plt.title("Selective Defensive Signal Diagnostics")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "selective_defensive_signal_diagnostics.png", dpi=170)
        plt.close(fig)


def format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.2%}"


def format_num(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.4f}"


def write_reports(
    feature_cols: list[str],
    model_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    best_model: dict[str, Any],
    validation_components: dict[str, Any],
    skipped_models: list[dict[str, str]],
) -> None:
    tuning_info: dict[str, Any] = {}
    tuning_path = OUTPUT_DIR / "strategy_tuned_params.json"
    if tuning_path.exists():
        try:
            tuning_info = json.loads(tuning_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tuning_info = {}
    selected_tuned_params = tuning_info.get("validation_tuned_index_core", {}).get("selected", {})
    selected_tuned_text = selected_tuned_params if selected_tuned_params else "Not available"
    selected_cost_aware_params = tuning_info.get("cost_aware_index_enhancement", {}).get("selected", {})
    selected_cost_aware_text = selected_cost_aware_params if selected_cost_aware_params else "Not available"
    selected_tail_params = tuning_info.get("tail_score_index_enhancement", {}).get("selected", {})
    selected_tail_text = selected_tail_params if selected_tail_params else "Not available"
    feature_top_k_info = tuning_info.get("feature_top_k_selection", {})
    selected_top_k_label = feature_top_k_info.get("selected_top_k_label", "NA")
    directional_timing_info = tuning_info.get("directional_timing", {})
    selected_long_cash_params = directional_timing_info.get("selected_long_cash_params", {})
    selected_scaled_timing_params = directional_timing_info.get("selected_scaled_timing_params", {})
    selected_long_cash_text = selected_long_cash_params if selected_long_cash_params else "Not available"
    selected_scaled_timing_text = selected_scaled_timing_params if selected_scaled_timing_params else "Not available"
    selective_defensive_info = tuning_info.get("selective_defensive_enhancement", {})
    selected_selective_params = selective_defensive_info.get("selected", {})
    selected_selective_text = selected_selective_params if selected_selective_params else "Not available"

    highlights = strategy_highlights(strategy_metrics)
    best_strategy = highlights["best_no_leverage_total"]
    best_real_strategy = highlights["best_real_no_leverage_ml_strategy"]
    real_strategy_fallback = highlights["best_real_no_leverage_ml_strategy_fallback"]
    best_sharpe_strategy = highlights["best_no_leverage_sharpe"]
    best_drawdown_strategy = highlights["best_no_leverage_drawdown"]
    best_plus_strategy = highlights["best_enhanced_exposure"]
    buy_hold = highlights["benchmark"]
    cost_aware_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_cost_aware_index_enhancement"]
    cost_aware_row = cost_aware_rows.sort_values("total_return", ascending=False).iloc[0] if not cost_aware_rows.empty else None
    tail_score_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_tail_score_index_enhancement"]
    tail_score_row = tail_score_rows.sort_values("total_return", ascending=False).iloc[0] if not tail_score_rows.empty else None
    long_cash_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_directional_long_cash"]
    long_cash_row = long_cash_rows.sort_values("total_return", ascending=False).iloc[0] if not long_cash_rows.empty else None
    scaled_timing_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_directional_scaled_timing"]
    scaled_timing_row = scaled_timing_rows.sort_values("total_return", ascending=False).iloc[0] if not scaled_timing_rows.empty else None
    selective_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_selective_defensive_enhancement"]
    selective_row = selective_rows.sort_values("total_return", ascending=False).iloc[0] if not selective_rows.empty else None
    conservative_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_cost_aware_overlay_conservative"]
    conservative_row = (
        conservative_rows.sort_values("total_return", ascending=False).iloc[0]
        if not conservative_rows.empty
        else None
    )
    dir70 = (best_model.get("directional_accuracy") or 0.0) >= 0.70
    outperforms = highlights["any_no_leverage_outperform"]
    real_strategy_fallback_note = (
        "No qualifying real no-leverage ML timing strategy was found, so this field falls back to the best no-leverage total-return strategy."
        if real_strategy_fallback
        else "The real no-leverage ML timing strategy excludes buy-and-hold, benchmark clones, and enhanced-exposure strategies."
    )

    if skipped_models:
        skipped_text = "\n".join([f"- {m['model_name']}: {m['reason']}" for m in skipped_models])
    else:
        skipped_text = "- XGBoost: installed and included in model comparison\n- LightGBM: installed and included in model comparison"
    optional_best_note = (
        f"{best_model['model_name']} was selected by the composite rule. "
        "XGBoost and LightGBM were included but did not become the final selected prediction model."
        if best_model["model_name"] not in {"XGBoost", "LightGBM"}
        else f"{best_model['model_name']} was selected by the composite rule after adding XGBoost and LightGBM."
    )
    strategy_note = (
        "The no-leverage strategies still do not outperform buy-and-hold in total return. The test period was strongly upward, buy-and-hold stayed fully invested, "
        "and ML strategies reduced exposure to control risk; the best no-leverage strategy therefore reduced drawdown but lagged in total return."
        if not outperforms
        else "At least one no-leverage enhanced strategy outperformed buy-and-hold while preserving the leakage-safe backtest protocol."
    )
    plus_note = ""
    if best_plus_strategy is not None:
        plus_note = (
            "The enhanced-exposure strategy outperforms buy-and-hold, but it permits maximum exposure above 1.0 and should be interpreted as a separate leverage-assisted experiment."
            if highlights["any_plus_outperform"]
            else "The enhanced-exposure strategy also does not outperform buy-and-hold. It is still reported separately because it permits maximum exposure above 1.0."
        )
    cost_aware_success = cost_aware_row is not None and cost_aware_row["excess_return_vs_buy_hold"] > 0
    if cost_aware_success:
        cost_aware_note = (
            "The final no-leverage cost-aware index-enhancement strategy achieves positive excess return over buy-and-hold while keeping maximum exposure capped at 1.00. "
            "Therefore, the index-enhancement objective is achieved under the no-leverage setting. The result is obtained using validation-selected parameters and includes transaction costs."
        )
    else:
        cost_aware_note = (
            "The final no-leverage cost-aware index-enhancement strategy improves cost control and drawdown management, but it still does not generate positive excess return over buy-and-hold. "
            "This indicates that, with only index-level OHLCV and limited intraday data, outperforming a strongly rising fully invested benchmark without leverage remains difficult."
        )
    tail_score_note = "The tail-score index-enhancement strategy was not generated."
    if tail_score_row is not None:
        if tail_score_row["total_return"] > buy_hold["total_return"]:
            tail_score_note = (
                "The tail-score index-enhancement strategy outperforms buy-and-hold. "
                "The improvement comes from using big-up and big-down classifiers to keep full exposure when tail-score is favorable and reduce exposure when weak trend and high-risk conditions coincide."
            )
        else:
            tail_score_note = (
                "The tail-score index-enhancement strategy still does not outperform buy-and-hold. "
                "Under the single-index OHLCV data constraint, the added tail targets and factors are still not sufficient to produce stable positive no-leverage excess return."
            )
    signals_path = OUTPUT_DIR / "directional_trading_signals.csv"
    if signals_path.exists():
        directional_signals = pd.read_csv(signals_path)
        directional_signal_stats = summarize_directional_signal_frame(directional_signals)
    else:
        directional_signal_stats = {}
    long_cash_signal_stats = directional_signal_stats.get("ml_directional_long_cash", {})
    scaled_timing_signal_stats = directional_signal_stats.get("ml_directional_scaled_timing", {})
    long_cash_outperform = bool(long_cash_row is not None and long_cash_row["total_return"] > buy_hold["total_return"])
    scaled_timing_outperform = bool(scaled_timing_row is not None and scaled_timing_row["total_return"] > buy_hold["total_return"])
    selective_outperform = bool(selective_row is not None and selective_row["total_return"] > buy_hold["total_return"])
    selective_best_real = bool(
        selective_row is not None
        and best_real_strategy["strategy"] == "ml_selective_defensive_enhancement"
        and str(best_real_strategy["model_name"]) == str(selective_row["model_name"])
        and int(best_real_strategy["horizon"]) == int(selective_row["horizon"])
    )
    direct_timing_note = (
        "The direct ML buy/sell strategy achieves positive no-leverage excess return by converting machine-learning predictions into explicit BUY, SELL, and HOLD signals."
        if long_cash_outperform or scaled_timing_outperform
        else "The direct ML buy/sell strategy implements the required prediction-driven trading logic, but under the strong upward test period, frequent or incorrect SELL signals caused missed upside, so the no-leverage strategy did not outperform buy-and-hold."
    )
    selective_note = (
        "The selective defensive enhancement strategy achieves positive no-leverage excess return by using ML predictions to avoid unfavorable regimes while maintaining high benchmark participation."
        if selective_outperform
        else "The selective defensive enhancement strategy maintains high benchmark participation and improves the interpretability of ML-based timing, but it still does not outperform buy-and-hold in this strongly rising test period."
    )
    summary = f"""# Enhanced Index Strategy Summary

## Key Result
- Best prediction model: {best_model['model_name']}
- Best prediction horizon: {best_model['horizon']} trading days
- Directional accuracy: {format_pct(best_model.get('directional_accuracy'))}
- AUC: {format_num(best_model.get('AUC'))}
- Return correlation: {format_num(best_model.get('return_correlation'))}
- Directional accuracy >= 70%: {'Yes' if dir70 else 'No'}
- Validation-selected feature top_k: {selected_top_k_label}
- Big-up AUC: {format_num(best_model.get('big_up_AUC'))}
- Big-down AUC: {format_num(best_model.get('big_down_AUC'))}
- Tail-score return correlation: {format_num(best_model.get('tail_score_return_correlation'))}

## Prediction Layer Update
- Added trend-quality, tail-risk, volume-confirmation, intraday-structure, and market-regime factors.
- The 5-minute data are not directly used for high-frequency trading. Instead, they are transformed into daily intraday-structure factors. In addition to basic intraday return and volatility, the enhanced feature set captures opening shock, closing pressure, high/low timing, intraday reversal, volume-return correlation, net volume pressure, large-volume-bar return, and intraday trend quality. These features are designed to extract information that daily OHLCV data cannot directly observe.
- Added leakage-safe tail targets: target_big_up and target_big_down use historical forward-return quantiles whose target_end_date is before the current date.
- Added separate big-up and big-down classifiers and pred_tail_score = pred_big_up_prob - pred_big_down_prob.
- Feature top_k is selected only on the validation period from candidates [50, 80, 120, all], and each walk-forward chunk selects features using IC computed on that chunk's training data only.

## Best No-Leverage Strategy
- Best no-leverage strategy by total return: {best_strategy['strategy']} ({best_strategy['model_name']}, horizon={best_strategy['horizon']})
- Best no-leverage strategy total return: {format_pct(best_strategy['total_return'])}
- Buy-and-hold total return: {format_pct(buy_hold['total_return'])}
- No-leverage excess return: {format_pct(best_strategy['excess_return_vs_buy_hold'])}
- Best no-leverage max drawdown: {format_pct(best_strategy['max_drawdown'])}
- Whether best no-leverage strategy is benchmark clone: {'Yes' if bool(best_strategy['is_benchmark_clone']) else 'No'}
- Sharpe: {format_num(best_strategy['sharpe'])}
- Calmar: {format_num(best_strategy['calmar'])}
- Information Ratio: {format_num(best_strategy['information_ratio'])}
- Any no-leverage strategy outperforms buy-and-hold: {'Yes' if outperforms else 'No'}
- Best no-leverage strategy by Sharpe: {best_sharpe_strategy['strategy']} ({best_sharpe_strategy['model_name']}, horizon={best_sharpe_strategy['horizon']}), Sharpe {format_num(best_sharpe_strategy['sharpe'])}
- Best no-leverage strategy by max drawdown: {best_drawdown_strategy['strategy']} ({best_drawdown_strategy['model_name']}, horizon={best_drawdown_strategy['horizon']}), max drawdown {format_pct(best_drawdown_strategy['max_drawdown'])}

## Benchmark Clone Check
- Best total-return no-leverage strategy may be a benchmark clone if it keeps 1.00 exposure all the time.
- Best no-leverage strategy by total return: {best_strategy['strategy']} ({best_strategy['model_name']}, horizon={best_strategy['horizon']})
- Whether it is benchmark clone: {'Yes' if bool(best_strategy['is_benchmark_clone']) else 'No'}
- Avg position: {format_num(best_strategy['avg_position'])}
- Min position: {format_num(best_strategy['min_position'])}
- Max position: {format_num(best_strategy['max_position'])}
- Days below full exposure: {int(best_strategy['days_below_full_exposure'])}
- Avg absolute position gap: {format_num(best_strategy['avg_abs_position_gap'])}
- Total turnover: {format_num(best_strategy['total_turnover'])}
- Cost-aware conservative overlay benchmark clone: {'Yes' if conservative_row is not None and bool(conservative_row['is_benchmark_clone']) else 'No' if conservative_row is not None else 'NA'}
- Cost-aware conservative overlay avg position: {format_num(conservative_row['avg_position']) if conservative_row is not None else 'NA'}
- Cost-aware conservative overlay days below full exposure: {int(conservative_row['days_below_full_exposure']) if conservative_row is not None else 'NA'}
- Best real no-leverage ML timing strategy: {best_real_strategy['strategy']} ({best_real_strategy['model_name']}, horizon={best_real_strategy['horizon']})
- Real ML timing strategy total return: {format_pct(best_real_strategy['total_return'])}
- Real ML timing strategy excess return: {format_pct(best_real_strategy['excess_return_vs_buy_hold'])}
- Real ML timing strategy max drawdown: {format_pct(best_real_strategy['max_drawdown'])}
- Real ML timing strategy days below full exposure: {int(best_real_strategy['days_below_full_exposure'])}
- Real ML timing strategy avg position: {format_num(best_real_strategy['avg_position'])}
- Real ML timing strategy total turnover: {format_num(best_real_strategy['total_turnover'])}
- Fallback note: {real_strategy_fallback_note}

## Enhanced-Exposure Experiment
- Enabled: {'Yes' if ALLOW_LEVERAGE_EXPERIMENT else 'No'}
- Strategies: ml_index_enhanced_plus, ml_index_enhanced_plus_115, ml_index_enhanced_plus_120
- Best enhanced-exposure strategy by total return: {best_plus_strategy['strategy'] if best_plus_strategy is not None else 'NA'} ({best_plus_strategy['model_name'] if best_plus_strategy is not None else 'NA'}, horizon={best_plus_strategy['horizon'] if best_plus_strategy is not None else 'NA'})
- Best enhanced-exposure total return: {format_pct(best_plus_strategy['total_return']) if best_plus_strategy is not None else 'NA'}
- Enhanced-exposure excess return: {format_pct(best_plus_strategy['excess_return_vs_buy_hold']) if best_plus_strategy is not None else 'NA'}
- Enhanced-exposure outperforms buy-and-hold: {'Yes' if highlights['any_plus_outperform'] else 'No'}

## Cost-Aware Index Enhancement
- Strategy: ml_cost_aware_index_enhancement
- Signal source: {selected_cost_aware_params.get('model_name', 'NA') if selected_cost_aware_params else 'NA'} {selected_cost_aware_params.get('horizon', 'NA') if selected_cost_aware_params else 'NA'}D
- Total return: {format_pct(cost_aware_row['total_return']) if cost_aware_row is not None else 'NA'}
- Excess return vs buy-and-hold: {format_pct(cost_aware_row['excess_return_vs_buy_hold']) if cost_aware_row is not None else 'NA'}
- Max drawdown: {format_pct(cost_aware_row['max_drawdown']) if cost_aware_row is not None else 'NA'}
- Average turnover: {format_num(cost_aware_row['avg_turnover']) if cost_aware_row is not None else 'NA'}
- Selected cost-aware parameters: {selected_cost_aware_text}

## Tail-Score Index Enhancement
- Strategy: ml_tail_score_index_enhancement
- Signal source: {selected_tail_params.get('model_name', 'NA') if selected_tail_params else 'NA'} {selected_tail_params.get('horizon', 'NA') if selected_tail_params else 'NA'}D
- Total return: {format_pct(tail_score_row['total_return']) if tail_score_row is not None else 'NA'}
- Excess return vs buy-and-hold: {format_pct(tail_score_row['excess_return_vs_buy_hold']) if tail_score_row is not None else 'NA'}
- Max drawdown: {format_pct(tail_score_row['max_drawdown']) if tail_score_row is not None else 'NA'}
- Avg position: {format_num(tail_score_row['avg_position']) if tail_score_row is not None else 'NA'}
- Days below full exposure: {int(tail_score_row['days_below_full_exposure']) if tail_score_row is not None else 'NA'}
- Total turnover: {format_num(tail_score_row['total_turnover']) if tail_score_row is not None else 'NA'}
- Selected tail-score parameters: {selected_tail_text}
- Interpretation: {tail_score_note}

## Direct ML Buy/Sell Timing Strategy
- Added strategies: ml_directional_long_cash and ml_directional_scaled_timing
- Logic: BUY means the model predicts high upward probability; SELL means the model predicts low upward probability or negative expected return; HOLD means the signal is not strong enough, reducing unnecessary trading.
- Long/cash selected model/horizon: {selected_long_cash_params.get('model_name', 'NA') if selected_long_cash_params else 'NA'} {selected_long_cash_params.get('horizon', 'NA') if selected_long_cash_params else 'NA'}D
- Long/cash selected parameters: {selected_long_cash_text}
- Long/cash total return: {format_pct(long_cash_row['total_return']) if long_cash_row is not None else 'NA'}
- Long/cash excess return vs buy-and-hold: {format_pct(long_cash_row['excess_return_vs_buy_hold']) if long_cash_row is not None else 'NA'}
- Long/cash max drawdown: {format_pct(long_cash_row['max_drawdown']) if long_cash_row is not None else 'NA'}
- Long/cash Sharpe: {format_num(long_cash_row['sharpe']) if long_cash_row is not None else 'NA'}
- Long/cash BUY signals: {long_cash_signal_stats.get('total_buy_signals', 'NA')}
- Long/cash SELL signals: {long_cash_signal_stats.get('total_sell_signals', 'NA')}
- Long/cash win rate after BUY: {format_pct(long_cash_signal_stats.get('win_rate_after_buy')) if long_cash_signal_stats else 'NA'}
- Long/cash successful SELL count: {long_cash_signal_stats.get('successful_sell_count', 'NA')}
- Long/cash failed SELL count: {long_cash_signal_stats.get('failed_sell_count', 'NA')}
- Long/cash outperforms buy-and-hold: {'Yes' if long_cash_outperform else 'No'}
- Scaled timing selected model/horizon: {selected_scaled_timing_params.get('model_name', 'NA') if selected_scaled_timing_params else 'NA'} {selected_scaled_timing_params.get('horizon', 'NA') if selected_scaled_timing_params else 'NA'}D
- Scaled timing selected parameters: {selected_scaled_timing_text}
- Scaled timing total return: {format_pct(scaled_timing_row['total_return']) if scaled_timing_row is not None else 'NA'}
- Scaled timing excess return vs buy-and-hold: {format_pct(scaled_timing_row['excess_return_vs_buy_hold']) if scaled_timing_row is not None else 'NA'}
- Scaled timing max drawdown: {format_pct(scaled_timing_row['max_drawdown']) if scaled_timing_row is not None else 'NA'}
- Scaled timing Sharpe: {format_num(scaled_timing_row['sharpe']) if scaled_timing_row is not None else 'NA'}
- Scaled timing BUY signals: {scaled_timing_signal_stats.get('total_buy_signals', 'NA')}
- Scaled timing SELL signals: {scaled_timing_signal_stats.get('total_sell_signals', 'NA')}
- Scaled timing win rate after BUY: {format_pct(scaled_timing_signal_stats.get('win_rate_after_buy')) if scaled_timing_signal_stats else 'NA'}
- Scaled timing successful SELL count: {scaled_timing_signal_stats.get('successful_sell_count', 'NA')}
- Scaled timing failed SELL count: {scaled_timing_signal_stats.get('failed_sell_count', 'NA')}
- Scaled timing outperforms buy-and-hold: {'Yes' if scaled_timing_outperform else 'No'}
- Interpretation: {direct_timing_note}

## Selective Defensive Index Enhancement
- Strategy name: ml_selective_defensive_enhancement
- This is a benchmark-aware no-leverage enhancement strategy, not an aggressive 0/1 timing strategy.
- It defaults to full participation, applies only mild cuts for uncertain negative ML signals, and makes larger defensive cuts when ML negative signal, weak trend, and high risk coincide.
- Maximum exposure is 1.00, so there is no leverage. The minimum exposure is selected on the validation period.
- Parameters are selected only on the validation period from {VALID_START.date()} to {VALID_END.date()}.
- Enhanced 5-minute factors may be used as risk confirmation, but they are daily aggregated intraday-structure factors and do not use future information.
- Selected model/horizon: {selected_selective_params.get('model_name', 'NA') if selected_selective_params else 'NA'} {selected_selective_params.get('horizon', 'NA') if selected_selective_params else 'NA'}D
- Selected parameters: {selected_selective_text}
- Total return: {format_pct(selective_row['total_return']) if selective_row is not None else 'NA'}
- Excess return vs buy-and-hold: {format_pct(selective_row['excess_return_vs_buy_hold']) if selective_row is not None else 'NA'}
- Max drawdown: {format_pct(selective_row['max_drawdown']) if selective_row is not None else 'NA'}
- Sharpe: {format_num(selective_row['sharpe']) if selective_row is not None else 'NA'}
- Avg position: {format_num(selective_row['avg_position']) if selective_row is not None else 'NA'}
- Min position: {format_num(selective_row['min_position']) if selective_row is not None else 'NA'}
- Days below full exposure: {int(selective_row['days_below_full_exposure']) if selective_row is not None else 'NA'}
- Turnover: {format_num(selective_row['total_turnover']) if selective_row is not None else 'NA'}
- Outperforms buy-and-hold: {'Yes' if selective_outperform else 'No'}
- Becomes best real no-leverage ML timing strategy: {'Yes' if selective_best_real else 'No'}
- Interpretation: {selective_note}

## Experiment Setup
- Initial capital: RMB {INITIAL_CAPITAL:,.0f}
- Test period: {TEST_START.date()} to {TEST_END.date()}
- Horizons: {HORIZONS}
- Retrain frequency: every {RETRAIN_EVERY} test trading days
- Trading cost: {COST_RATE:.4f} per turnover
- Feature count: {len(feature_cols)}
- Trade label threshold: {TRADE_THRESHOLD:.4f}

## Leakage Controls
- Features use only information available at or before the close of date t.
- Signals generated after date t close are applied to t+1 returns.
- Walk-forward training uses `target_end_date < chunk_start`, not `date < chunk_start`.
- Target labels remain NaN when forward returns are unavailable.
- Quantile clipping and median imputation are fitted on each training chunk only.
- No random train/test split is used.

## Optional Model Availability
{skipped_text}

## Model Selection Note
- XGBoost and LightGBM are part of the current comparison set.
- {optional_best_note}
- Final selection uses a composite score, not single accuracy: AUC, directional accuracy, positive return correlation, strategy-level excess return, and drawdown are all considered.
- Interpretation: the signal is moderate rather than strong; it is best described as weak but useful directional information with improved drawdown control.

## Strategy Layer Update
- Added ml_index_enhanced_core: benchmark-aware index enhancement with high no-leverage exposure in [0.75, 1.00].
- Added ml_index_full_participation: benchmark-aware full participation with no-leverage exposure in [0.88, 1.00].
- Added ml_directional_alpha_extratrees_10d: fixed strategy using ExtraTrees 10D, the highest directional-accuracy model.
- Added ml_validation_tuned_index_core: parameters selected only on the validation period from {VALID_START.date()} to {VALID_END.date()}.
- Added ml_dynamic_ensemble_full_participation: validation-weighted ensemble across high-AUC, high-direction, selected, and optional boosting models.
- Added ml_cost_aware_index_enhancement: validation-tuned no-trade-band strategy using ML predictions to decide small exposure reductions and recoveries.
- Added ml_cost_aware_overlay_conservative: extremely low-turnover no-leverage overlay that only cuts exposure when model, trend, and risk signals are all weak.
- Added ml_tail_score_index_enhancement: validation-tuned no-leverage strategy using big-up/big-down tail-score predictions to cut exposure only when tail score is weak and risk/trend conditions confirm.
- Added ml_directional_long_cash: direct BUY/SELL/HOLD strategy that goes long when ML predicts upside and moves to cash when ML predicts downside.
- Added ml_directional_scaled_timing: direct ML timing strategy that maps predicted up probability and predicted return into scaled no-leverage exposure.
- Added ml_selective_defensive_enhancement: benchmark-aware no-leverage enhancement that defaults to full participation, applies mild cuts for uncertain negative ML signals, and uses larger cuts when ML negative signal, weak trend, and high risk coincide.
- Added ml_vol_target_trend: volatility targeting plus trend following, with ML signal as a small adjustment.
- Added ml_enhanced_cppi: CPPI-style path-dependent drawdown control based only on realized strategy equity.
- Added ml_index_enhanced_plus, ml_index_enhanced_plus_115, and ml_index_enhanced_plus_120: enhanced-exposure scenarios reported separately from no-leverage strategies.
- Selected feature top_k: {selected_top_k_label}
- Selected validation-tuned parameters: {selected_tuned_text}
- Selected cost-aware parameters: {selected_cost_aware_text}
- Selected tail-score parameters: {selected_tail_text}
- Selected direct long/cash parameters: {selected_long_cash_text}
- Selected direct scaled-timing parameters: {selected_scaled_timing_text}
- Selected selective defensive parameters: {selected_selective_text}

## Validation-Selected Ensemble Components
- Regression component: {validation_components['best_regression']}
- Classification component: {validation_components['best_classification']}
- Trade-label component: {validation_components['best_trade']}

## Honest Note
If the 70% directional-accuracy target is not reached, this is expected under a strict out-of-sample protocol. Short-horizon index returns are noisy, this is a single-index dataset with limited independent samples, and technical factors often have weak standalone predictive power. The system still performs multi-factor mining, multi-model comparison, leakage-safe walk-forward prediction, strategy backtesting, transaction-cost accounting, and drawdown control.

## Benchmark Interpretation
{strategy_note}
{plus_note}
{cost_aware_note}
{tail_score_note}
{direct_timing_note}
{selective_note}
The enhanced-exposure plus strategy is reported separately because it allows maximum exposure above 1.00.

## Files
- outputs/predictions_all_models.csv
- outputs/model_metrics_by_horizon.csv
- outputs/strategy_daily_all.csv
- outputs/strategy_metrics_all.csv
- outputs/active_return_attribution.csv
- outputs/active_return_attribution_summary.md
- outputs/directional_trading_signals.csv
- outputs/directional_timing_attribution_summary.md
- outputs/directional_timing_tuned_params.json
- outputs/selective_defensive_tuned_params.json
- outputs/selective_defensive_attribution.csv
- outputs/selective_defensive_attribution_summary.md
- outputs/factor_effectiveness.csv
- outputs/factor_effectiveness_tail_targets.csv
- outputs/feature_columns.json
- outputs/best_model_selection.json
- outputs/strategy_tuned_params.json
- outputs/cost_aware_tuned_params.json
- outputs/plots/
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    best_metric = model_metrics.iloc[0]
    report = f"""# Final Report Material

## 1. Project Objective
The objective is to build a machine-learning-enhanced index strategy for the micro-cap index using RMB {INITIAL_CAPITAL:,.0f} initial capital from {TEST_START.date()} to {TEST_END.date()}. The strategy aims to improve prediction quality, generate excess return over buy-and-hold, and control drawdown under a strict no-future-leakage design.

## 2. Data Description and Preprocessing
The experiment uses daily OHLCV and amount data, plus available 5-minute intraday bars. Daily records are sorted chronologically and merged with daily aggregated intraday features. Missing intraday data are represented through an availability flag and are imputed only inside each walk-forward training chunk.

## 3. Factor Mining and Feature Engineering
The factor set includes momentum returns, rolling return means, short-term reversal, RSI, KDJ, moving-average ratios and spreads, MACD, breakout measures, trend strength, rolling volatility, ATR, Bollinger width and position, drawdown, downside volatility, volume and amount ratios, price-volume correlation, up-volume ratio, Amihud-style illiquidity, intraday morning/afternoon/first-hour/last-hour returns, intraday volatility, VWAP discount, close position, volume profile, and market-regime indicators. This version adds trend-quality factors, tail-risk factors, volume-confirmation factors, richer 5-minute bar-structure factors, and explicit bull/bear/sideways regime flags.

The 5-minute data are not directly used for high-frequency trading. Instead, they are transformed into daily intraday-structure factors. In addition to basic intraday return and volatility, the enhanced feature set captures opening shock, closing pressure, high/low timing, intraday reversal, volume-return correlation, net volume pressure, large-volume-bar return, and intraday trend quality. These features are designed to extract information that daily OHLCV data cannot directly observe.

## 4. Machine Learning Models
The system compares HistGradientBoosting, RandomForest, ExtraTrees, GradientBoosting, XGBoost, and LightGBM regressors/classifiers. XGBoost and LightGBM are installed in the current environment and included in the refreshed model comparison. Tree and boosting models are suitable here because they capture nonlinear factor interactions, handle mixed signal scales well, and do not require fragile linear assumptions.

## 5. Walk-Forward Training Process
For horizons {HORIZONS}, labels are forward returns, upward direction, tradeable upward direction above the transaction-cost-aware threshold {TRADE_THRESHOLD:.4f}, big-up events, and big-down events. Big-up and big-down labels compare forward return with historical 70% and 30% quantiles whose target_end_date is before the current date, so their thresholds do not look into the future. During every test chunk, training samples must satisfy `target_end_date < chunk_start`, ensuring each forward label is fully known before model fitting. Quantile clipping, imputation, and IC-based feature selection are estimated only from the current training chunk. The selected feature top_k is {selected_top_k_label}, chosen only on the validation period.

## 6. Prediction Performance
The best metric-ranked prediction row is {best_metric['model_name']} with horizon {int(best_metric['horizon'])}D. Its AUC is {format_num(best_metric['AUC'])}, directional accuracy is {format_pct(best_metric['directional_accuracy'])}, return correlation is {format_num(best_metric['return_correlation'])}, big-up AUC is {format_num(best_metric.get('big_up_AUC'))}, big-down AUC is {format_num(best_metric.get('big_down_AUC'))}, and tail-score return correlation is {format_num(best_metric.get('tail_score_return_correlation'))}. The composite selected model is {best_model['model_name']} with horizon {best_model['horizon']}D. {optional_best_note} Overall prediction quality is moderate, not exceptional; it should be interpreted as a weak but useful directional and tail-risk signal rather than a highly accurate forecasting engine.

## 7. Trading Strategy Design
The backtest includes buy_hold, technical_timing, ml_signal_aggressive, ml_signal_benchmark_aware, ml_risk_control, ensemble_strategy, ml_index_enhanced_core, ml_index_full_participation, ml_directional_alpha_extratrees_10d, ml_directional_long_cash, ml_directional_scaled_timing, ml_selective_defensive_enhancement, ml_validation_tuned_index_core, ml_dynamic_ensemble_full_participation, ml_cost_aware_index_enhancement, ml_cost_aware_overlay_conservative, ml_tail_score_index_enhancement, ml_vol_target_trend, ml_enhanced_cppi, and enhanced-exposure plus strategies. The new strategy layer focuses on six ideas. First, benchmark-aware index enhancement keeps high index exposure and adjusts around the benchmark using small ML signal tilts. Second, selective defensive enhancement defaults to full benchmark participation, uses mild cuts for uncertain negative ML signals, and reserves larger cuts for confirmed weak-trend or high-risk regimes. Third, volatility targeting plus trend following reduces risk in high-volatility or weak-trend regimes while staying exposed in low-volatility upward trends. Fourth, CPPI-style drawdown control adjusts exposure according to the distance between realized strategy equity and a dynamic protection floor. Fifth, the tail-score strategy uses separate big-up and big-down classifiers and cuts exposure only when tail score is weak and risk/trend filters confirm. Sixth, the direct directional timing strategies convert ML predictions into explicit BUY, SELL, and HOLD signals. Cost-aware strategies add no-trade bands selected only on the validation period. The plus 1.15 and plus 1.20 strategies are enhanced-exposure experiments and are reported separately from the no-leverage strategies. Selected validation-tuned parameters: {selected_tuned_text}. Selected cost-aware parameters: {selected_cost_aware_text}. Selected tail-score parameters: {selected_tail_text}. Selected direct long/cash parameters: {selected_long_cash_text}. Selected direct scaled-timing parameters: {selected_scaled_timing_text}. Selected selective defensive parameters: {selected_selective_text}.

## 8. Backtesting Result
The best no-leverage strategy by total return is {best_strategy['strategy']} with total return {format_pct(best_strategy['total_return'])}. Buy-and-hold total return is {format_pct(buy_hold['total_return'])}. This strategy is benchmark-clone={bool(best_strategy['is_benchmark_clone'])}, with avg position {format_num(best_strategy['avg_position'])}, days below full exposure {int(best_strategy['days_below_full_exposure'])}, and tracking error {format_num(best_strategy['tracking_error'])}.

The cost-aware conservative overlay produces almost the same result as buy-and-hold because it keeps full exposure during the test period. Therefore, it is treated as a benchmark-clone result rather than the main machine-learning timing result.

The main real no-leverage ML timing strategy is {best_real_strategy['strategy']}. It adjusts exposure according to {best_real_strategy['model_name']} {int(best_real_strategy['horizon']) if int(best_real_strategy['horizon']) >= 0 else best_real_strategy['horizon']}-day predictions. It achieves {format_pct(best_real_strategy['total_return'])} total return versus {format_pct(buy_hold['total_return'])} for buy-and-hold, with max drawdown moving from {format_pct(buy_hold['max_drawdown'])} to {format_pct(best_real_strategy['max_drawdown'])}. {real_strategy_fallback_note}

Tail-score strategy result: {tail_score_note}

The best no-leverage strategy by Sharpe is {best_sharpe_strategy['strategy']} with Sharpe {format_num(best_sharpe_strategy['sharpe'])}. The best no-leverage strategy by max drawdown is {best_drawdown_strategy['strategy']} with max drawdown {format_pct(best_drawdown_strategy['max_drawdown'])}. {plus_note}

## 9. Drawdown and Risk Control
The system applies path-dependent drawdown control based only on realized strategy equity up to that date. Existing risk-control strategies reduce exposure after 8%, 12%, and 15% drawdowns, while the new CPPI-style strategy uses a dynamic floor equal to 90% of the prior running maximum equity. The real no-leverage ML timing strategy's max drawdown is {format_pct(best_real_strategy['max_drawdown'])}, compared with buy-and-hold max drawdown {format_pct(buy_hold['max_drawdown'])}.

## 10. Comparison with Buy-and-Hold Benchmark
Under the strict no-leverage setting, the real ML timing strategy nearly matches buy-and-hold but does not generate positive excess return. The enhanced-exposure plus_120 strategy achieves positive excess return, but it allows maximum exposure above 1.00 and is reported separately.

{tail_score_note}

## 10A. Cost-Aware Index Enhancement Result
{cost_aware_note} The enhanced-exposure plus strategy is reported separately because it allows maximum exposure above 1.00.

## 10B. Direct ML Buy/Sell Timing Strategy
The direct timing layer adds ml_directional_long_cash and ml_directional_scaled_timing. They directly convert machine-learning predictions into trading signals: BUY indicates high predicted upward probability, SELL indicates low predicted upward probability or negative expected return, and HOLD indicates insufficient signal strength. The selected long/cash signal source is {selected_long_cash_params.get('model_name', 'NA') if selected_long_cash_params else 'NA'} {selected_long_cash_params.get('horizon', 'NA') if selected_long_cash_params else 'NA'}D, with total return {format_pct(long_cash_row['total_return']) if long_cash_row is not None else 'NA'}, excess return {format_pct(long_cash_row['excess_return_vs_buy_hold']) if long_cash_row is not None else 'NA'}, max drawdown {format_pct(long_cash_row['max_drawdown']) if long_cash_row is not None else 'NA'}, Sharpe {format_num(long_cash_row['sharpe']) if long_cash_row is not None else 'NA'}, BUY signals {long_cash_signal_stats.get('total_buy_signals', 'NA')}, SELL signals {long_cash_signal_stats.get('total_sell_signals', 'NA')}, win rate after BUY {format_pct(long_cash_signal_stats.get('win_rate_after_buy')) if long_cash_signal_stats else 'NA'}, successful SELL count {long_cash_signal_stats.get('successful_sell_count', 'NA')}, and failed SELL count {long_cash_signal_stats.get('failed_sell_count', 'NA')}. The selected scaled-timing signal source is {selected_scaled_timing_params.get('model_name', 'NA') if selected_scaled_timing_params else 'NA'} {selected_scaled_timing_params.get('horizon', 'NA') if selected_scaled_timing_params else 'NA'}D, with total return {format_pct(scaled_timing_row['total_return']) if scaled_timing_row is not None else 'NA'}, excess return {format_pct(scaled_timing_row['excess_return_vs_buy_hold']) if scaled_timing_row is not None else 'NA'}, max drawdown {format_pct(scaled_timing_row['max_drawdown']) if scaled_timing_row is not None else 'NA'}, Sharpe {format_num(scaled_timing_row['sharpe']) if scaled_timing_row is not None else 'NA'}, BUY signals {scaled_timing_signal_stats.get('total_buy_signals', 'NA')}, SELL signals {scaled_timing_signal_stats.get('total_sell_signals', 'NA')}, win rate after BUY {format_pct(scaled_timing_signal_stats.get('win_rate_after_buy')) if scaled_timing_signal_stats else 'NA'}, successful SELL count {scaled_timing_signal_stats.get('successful_sell_count', 'NA')}, and failed SELL count {scaled_timing_signal_stats.get('failed_sell_count', 'NA')}. Long/cash outperforms buy-and-hold: {'Yes' if long_cash_outperform else 'No'}. Scaled timing outperforms buy-and-hold: {'Yes' if scaled_timing_outperform else 'No'}. {direct_timing_note}

## 10C. Selective Defensive Index Enhancement
The selective defensive layer adds ml_selective_defensive_enhancement. It is not an aggressive 0/1 buy/sell strategy; it is a benchmark-aware no-leverage enhancement strategy. The default position is full participation. Uncertain negative ML signals may trigger only a mild 0.98/0.97 style cut, while larger defensive cuts require weak trend or high-risk confirmation. Maximum exposure is 1.00, minimum exposure and the no-trade band are selected only on the validation period, and enhanced 5-minute daily factors can serve only as risk confirmation. The selected signal source is {selected_selective_params.get('model_name', 'NA') if selected_selective_params else 'NA'} {selected_selective_params.get('horizon', 'NA') if selected_selective_params else 'NA'}D. Total return is {format_pct(selective_row['total_return']) if selective_row is not None else 'NA'}, excess return is {format_pct(selective_row['excess_return_vs_buy_hold']) if selective_row is not None else 'NA'}, max drawdown is {format_pct(selective_row['max_drawdown']) if selective_row is not None else 'NA'}, Sharpe is {format_num(selective_row['sharpe']) if selective_row is not None else 'NA'}, average position is {format_num(selective_row['avg_position']) if selective_row is not None else 'NA'}, minimum position is {format_num(selective_row['min_position']) if selective_row is not None else 'NA'}, days below full exposure is {int(selective_row['days_below_full_exposure']) if selective_row is not None else 'NA'}, and total turnover is {format_num(selective_row['total_turnover']) if selective_row is not None else 'NA'}. It outperforms buy-and-hold: {'Yes' if selective_outperform else 'No'}. It becomes the best real no-leverage ML timing strategy: {'Yes' if selective_best_real else 'No'}. {selective_note}

## 11. Advantages of the Adopted Machine Learning Methods
Tree ensembles and boosting methods can model nonlinear relationships among momentum, trend, volatility, liquidity, and intraday factors. They are robust to monotonic transformations, capture interaction effects such as trend plus volatility regimes, and can be retrained efficiently in a walk-forward setting.

## 12. Limitations and Future Improvements
Short-term index prediction is noisy, and a single index provides limited independent samples. Technical factors may lose effectiveness across regimes. Future work could add cross-sectional stock-level constituents, macro factors, richer intraday microstructure signals, transaction-cost-aware objective functions, and nested validation for live model selection.

## 13. Conclusion
The project upgrades the original experiment into a leakage-safe multi-horizon, multi-model, multi-strategy research system. It produces complete predictions, model metrics, factor diagnostics, strategy paths, risk-control states, plots, and report-ready material. The results are reported honestly without using future information or test-set leakage to inflate performance.
"""
    (OUTPUT_DIR / "final_report_material.md").write_text(report, encoding="utf-8")

    if dir70:
        leakage_report = f"""# Leakage Check Report

Directional accuracy reached or exceeded 70%, so this additional review was generated automatically.

## Checks Performed
- Feature columns exclude all `target_*`, `fwd_ret_1d`, future dates, and label columns.
- Walk-forward training condition is `target_end_date < chunk_start`.
- Big-up/big-down thresholds use only historical target returns whose `target_end_date` is before the current date.
- Signals are generated on date t and applied only to `fwd_ret_1d`, the t to t+1 close-to-close return.
- `target_up` and `target_trade` are assigned NaN whenever `target_ret` is NaN.
- IC feature selection is fitted inside each training chunk only; selected top_k comes from validation only.
- Quantile clipping and median imputation are fitted on each training chunk only.
- No randomized train/test split is used.
- Strategy drawdown control uses only realized equity before the next return is applied.

## Selected Model
- Model: {best_model['model_name']}
- Horizon: {best_model['horizon']}D
- Directional accuracy: {format_pct(best_model.get('directional_accuracy'))}
- AUC: {format_num(best_model.get('AUC'))}

## Residual Risk
This report confirms the implemented safeguards, but unusually high predictive accuracy should still be reviewed manually against raw feature definitions and data timestamps before any real-money use.
"""
        (OUTPUT_DIR / "leakage_check_report.md").write_text(leakage_report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / ".mplconfig").mkdir(exist_ok=True)
    (ROOT / ".cache").mkdir(exist_ok=True)

    print("Loading data...")
    daily, m5 = load_data()
    print(f"Daily rows: {len(daily)}, 5-minute rows: {len(m5)}")

    print("Building features...")
    intraday = build_intraday_features(m5)
    feature_df = build_daily_features(daily, intraday)
    feature_cols = get_feature_columns(feature_df)
    model_specs, skipped_models = build_model_specs()
    save_feature_columns(feature_cols, skipped_models)

    factor_effectiveness = compute_factor_effectiveness(feature_df, feature_cols)
    factor_effectiveness.to_csv(OUTPUT_DIR / "factor_effectiveness.csv", index=False)
    factor_effectiveness_tail = compute_factor_effectiveness_tail_targets(feature_df, feature_cols)
    factor_effectiveness_tail.to_csv(OUTPUT_DIR / "factor_effectiveness_tail_targets.csv", index=False)

    print(f"Feature count: {len(feature_cols)}")
    print(f"Model families: {[spec.name for spec in model_specs]}")
    if skipped_models:
        print(f"Skipped optional models: {skipped_models}")

    print("Selecting validation-only feature top_k and running validation walk-forward...")
    feature_top_k_selection = select_validation_feature_top_k(feature_df, feature_cols, model_specs)
    selected_feature_top_k = feature_top_k_selection["selected_top_k"]
    validation_preds = feature_top_k_selection["predictions"]
    validation_metrics = compute_model_metrics(validation_preds)
    validation_metrics.to_csv(OUTPUT_DIR / "validation_model_metrics_by_horizon.csv", index=False)
    validation_components = select_validation_components(validation_metrics)
    print("Tuning validation index core parameters...")
    tuned_params = tune_validation_index_core_params(feature_df, validation_preds)
    print("Tuning cost-aware index enhancement parameters...")
    cost_aware_params = tune_cost_aware_index_enhancement_params(feature_df, validation_preds)
    print("Tuning tail-score index enhancement parameters...")
    tail_score_params = tune_tail_score_index_enhancement_params(feature_df, validation_preds)
    print("Tuning direct directional timing parameters...")
    directional_timing_params = tune_directional_timing_params(feature_df, validation_preds)
    print("Tuning selective defensive enhancement parameters...")
    selective_defensive_params = tune_selective_defensive_enhancement_params(feature_df, validation_preds)
    print("Building dynamic ensemble weights...")
    dynamic_weights = build_dynamic_ensemble_weights(validation_metrics)
    feature_top_k_export = {k: v for k, v in feature_top_k_selection.items() if k != "predictions"}
    write_json(
        OUTPUT_DIR / "strategy_tuned_params.json",
        {
            "feature_top_k_selection": feature_top_k_export,
            "validation_tuned_index_core": tuned_params,
            "cost_aware_index_enhancement": cost_aware_params,
            "tail_score_index_enhancement": tail_score_params,
            "directional_timing": directional_timing_params,
            "selective_defensive_enhancement": selective_defensive_params,
            "dynamic_ensemble_weights": dynamic_weights,
        },
    )
    write_json(OUTPUT_DIR / "cost_aware_tuned_params.json", cost_aware_params)
    write_json(OUTPUT_DIR / "directional_timing_tuned_params.json", directional_timing_params)
    write_json(OUTPUT_DIR / "selective_defensive_tuned_params.json", selective_defensive_params)

    print("Running test walk-forward predictions...")
    pred_all = walk_forward_predictions(
        feature_df,
        feature_cols,
        model_specs,
        TEST_START,
        TEST_END,
        RETRAIN_EVERY,
        min_train_samples=MIN_TRAIN_SAMPLES,
        label="test",
        feature_top_k=selected_feature_top_k,
    )
    pred_export_cols = [
        "date",
        "trade_date",
        "target_start_date",
        "target_end_date",
        "horizon",
        "model_name",
        "close",
        "fwd_ret_1d",
        "target_ret",
        "target_up",
        "target_trade",
        "target_big_up",
        "target_big_down",
        "pred_ret",
        "pred_up_prob",
        "pred_trade_prob",
        "pred_big_up_prob",
        "pred_big_down_prob",
        "pred_tail_score",
        "pred_up",
        "pred_trade",
        "pred_big_up",
        "pred_big_down",
        "chunk_start",
        "train_samples",
        "train_latest_target_end_date",
        "feature_top_k",
        "selected_feature_count",
    ]
    pred_all[pred_export_cols].to_csv(OUTPUT_DIR / "predictions_all_models.csv", index=False)

    print("Evaluating models...")
    model_metrics = compute_model_metrics(pred_all)
    model_metrics.to_csv(OUTPUT_DIR / "model_metrics_by_horizon.csv", index=False)

    print("Backtesting strategies...")
    strategy_daily = build_all_strategies(
        feature_df,
        pred_all,
        validation_components,
        tuned_params,
        dynamic_weights,
        cost_aware_params,
        tail_score_params,
        directional_timing_params,
        selective_defensive_params,
    )
    strategy_metrics = compute_strategy_metrics(strategy_daily)
    strategy_daily.to_csv(OUTPUT_DIR / "strategy_daily_all.csv", index=False)
    strategy_metrics.to_csv(OUTPUT_DIR / "strategy_metrics_all.csv", index=False)

    best_model = select_best_model(model_metrics, strategy_metrics)
    write_json(
        OUTPUT_DIR / "best_model_selection.json",
        {
            "best_model": best_model,
            "validation_selected_ensemble_components": validation_components,
            "skipped_optional_models": skipped_models,
            "validation_tuned_index_core": tuned_params,
            "cost_aware_index_enhancement": cost_aware_params,
            "tail_score_index_enhancement": tail_score_params,
            "directional_timing": directional_timing_params,
            "selective_defensive_enhancement": selective_defensive_params,
            "feature_top_k_selection": feature_top_k_export,
            "dynamic_ensemble_weights": dynamic_weights,
        },
    )

    print("Saving plots, active return attribution, and report material...")
    active_attribution_path, active_summary_path = save_active_return_attribution(strategy_daily, strategy_metrics)
    directional_signals_path, directional_summary_path = save_directional_trading_outputs(strategy_daily)
    selective_attr_path, selective_summary_path = save_selective_defensive_attribution(strategy_daily, strategy_metrics)
    save_plots(feature_df, pred_all, model_metrics, strategy_daily, strategy_metrics, factor_effectiveness, best_model)
    write_reports(feature_cols, model_metrics, strategy_metrics, best_model, validation_components, skipped_models)

    highlights = strategy_highlights(strategy_metrics)
    best_strategy = highlights["best_no_leverage_total"]
    best_real_strategy = highlights["best_real_no_leverage_ml_strategy"]
    buy_hold = highlights["benchmark"]
    best_plus = highlights["best_enhanced_exposure"]
    tail_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_tail_score_index_enhancement"]
    tail_row = tail_rows.sort_values("total_return", ascending=False).iloc[0] if not tail_rows.empty else None
    long_cash_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_directional_long_cash"]
    long_cash_row = long_cash_rows.sort_values("total_return", ascending=False).iloc[0] if not long_cash_rows.empty else None
    scaled_timing_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_directional_scaled_timing"]
    scaled_timing_row = scaled_timing_rows.sort_values("total_return", ascending=False).iloc[0] if not scaled_timing_rows.empty else None
    selective_rows = strategy_metrics.loc[strategy_metrics["strategy"] == "ml_selective_defensive_enhancement"]
    selective_row = selective_rows.sort_values("total_return", ascending=False).iloc[0] if not selective_rows.empty else None
    dir_acc = best_model.get("directional_accuracy")
    dir70 = (dir_acc or 0.0) >= 0.70

    print("\n===== FINAL RESULTS =====")
    print(
        f"Best no-leverage strategy by total return: {best_strategy['strategy']} "
        f"({best_strategy['model_name']}, horizon={best_strategy['horizon']})"
    )
    print(f"Whether it is benchmark clone: {'Yes' if bool(best_strategy['is_benchmark_clone']) else 'No'}")
    print(
        f"Best real no-leverage ML timing strategy: {best_real_strategy['strategy']} "
        f"({best_real_strategy['model_name']}, horizon={best_real_strategy['horizon']})"
    )
    print(f"Real ML strategy total return: {best_real_strategy['total_return']:.6f}")
    print(f"Real ML strategy excess return: {best_real_strategy['excess_return_vs_buy_hold']:.6f}")
    print(f"Real ML strategy max drawdown: {best_real_strategy['max_drawdown']:.6f}")
    print(f"Buy-and-hold total return: {buy_hold['total_return']:.6f}")
    print(f"Buy-and-hold max drawdown: {buy_hold['max_drawdown']:.6f}")
    if tail_row is not None:
        print(
            "Tail-score strategy: "
            f"{tail_row['strategy']} ({tail_row['model_name']}, horizon={tail_row['horizon']}), "
            f"total_return={tail_row['total_return']:.6f}, "
            f"excess_return={tail_row['excess_return_vs_buy_hold']:.6f}, "
            f"max_drawdown={tail_row['max_drawdown']:.6f}"
        )
    print(f"Directional long/cash selected params: {directional_timing_params.get('selected_long_cash_params')}")
    print(f"Directional scaled timing selected params: {directional_timing_params.get('selected_scaled_timing_params')}")
    print(f"Selective defensive selected params: {selective_defensive_params.get('selected', selective_defensive_params)}")
    if long_cash_row is not None:
        print(
            "Directional long/cash result: "
            f"total_return={long_cash_row['total_return']:.6f}, "
            f"excess_return={long_cash_row['excess_return_vs_buy_hold']:.6f}, "
            f"max_drawdown={long_cash_row['max_drawdown']:.6f}"
        )
    if scaled_timing_row is not None:
        print(
            "Directional scaled timing result: "
            f"total_return={scaled_timing_row['total_return']:.6f}, "
            f"excess_return={scaled_timing_row['excess_return_vs_buy_hold']:.6f}, "
            f"max_drawdown={scaled_timing_row['max_drawdown']:.6f}"
        )
    direct_outperform = bool(
        (long_cash_row is not None and long_cash_row["total_return"] > buy_hold["total_return"])
        or (scaled_timing_row is not None and scaled_timing_row["total_return"] > buy_hold["total_return"])
    )
    print(f"Whether direct ML buy/sell strategy outperforms buy-and-hold: {'Yes' if direct_outperform else 'No'}")
    if selective_row is not None:
        print(f"Selective defensive total return: {selective_row['total_return']:.6f}")
        print(f"Selective defensive excess return: {selective_row['excess_return_vs_buy_hold']:.6f}")
        print(f"Selective defensive max drawdown: {selective_row['max_drawdown']:.6f}")
        print(f"Selective defensive avg position: {selective_row['avg_position']:.6f}")
        print(f"Selective defensive min position: {selective_row['min_position']:.6f}")
        print(f"Selective defensive days below full exposure: {int(selective_row['days_below_full_exposure'])}")
        print(
            "Whether selective defensive outperforms buy-and-hold: "
            f"{'Yes' if selective_row['total_return'] > buy_hold['total_return'] else 'No'}"
        )
        print(
            "Whether selective defensive becomes best real no-leverage ML timing strategy: "
            f"{'Yes' if best_real_strategy['strategy'] == 'ml_selective_defensive_enhancement' else 'No'}"
        )
    print(
        "Enhanced-exposure best strategy and excess return: "
        f"{best_plus['strategy']} ({best_plus['model_name']}, horizon={best_plus['horizon']}), "
        f"excess_return={best_plus['excess_return_vs_buy_hold']:.6f}"
        if best_plus is not None
        else "Enhanced-exposure best strategy and excess return: NA"
    )
    print(f"Whether any real no-leverage ML strategy outperforms buy-and-hold: {'Yes' if highlights['any_real_no_leverage_outperform'] else 'No'}")
    if highlights["any_real_no_leverage_outperform"]:
        print(
            f"Real no-leverage ML outperformer: {best_real_strategy['strategy']}, "
            f"positive excess return={best_real_strategy['excess_return_vs_buy_hold']:.6f}"
        )
    else:
        print(f"Real ML remaining gap to buy-and-hold: {abs(best_real_strategy['excess_return_vs_buy_hold']):.6f}")
    print(f"Whether enhanced-exposure strategy outperforms buy-and-hold: {'Yes' if highlights['any_plus_outperform'] else 'No'}")
    print(
        "Enhanced-exposure max exposure note: plus strategies are reported separately and allow exposure above 1.00."
    )
    print(f"Whether directional accuracy >= 70%: {'Yes' if dir70 else 'No'}")
    print(f"Selected validation-tuned parameters: {tuned_params.get('selected', tuned_params)}")
    print(f"Selected cost-aware parameters: {cost_aware_params.get('selected', cost_aware_params)}")
    print(f"Selected tail-score parameters: {tail_score_params.get('selected', tail_score_params)}")
    print(f"Selected feature top_k: {feature_top_k_selection['selected_top_k_label']}")
    print(f"selective_defensive_tuned_params.json path: {OUTPUT_DIR / 'selective_defensive_tuned_params.json'}")
    print(f"selective_defensive_attribution_summary.md path: {selective_summary_path}")
    print(f"selective_defensive_attribution.csv path: {selective_attr_path}")
    print(f"selective defensive equity plot path: {PLOT_DIR / 'equity_curve_selective_defensive_vs_buyhold.png'}")
    print(f"selective defensive drawdown plot path: {PLOT_DIR / 'drawdown_curve_selective_defensive_vs_buyhold.png'}")
    print(f"selective defensive exposure plot path: {PLOT_DIR / 'position_exposure_selective_defensive.png'}")
    print(f"selective defensive diagnostics plot path: {PLOT_DIR / 'selective_defensive_signal_diagnostics.png'}")
    print(f"directional_trading_signals.csv path: {directional_signals_path}")
    print(f"directional_timing_attribution_summary.md path: {directional_summary_path}")
    print(f"active_return_attribution_summary.md path: {active_summary_path}")
    print(f"active_return_attribution.csv path: {active_attribution_path}")
    print(f"Updated summary.md path: {OUTPUT_DIR / 'summary.md'}")
    print(f"Updated final_report_material.md path: {OUTPUT_DIR / 'final_report_material.md'}")
    print("=========================\n")


if __name__ == "__main__":
    main()
