from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

from .config import TAIL_QUANTILE_MIN_SAMPLES


try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


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


def finite_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def write_json_ready(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): write_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [write_json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [write_json_ready(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj
