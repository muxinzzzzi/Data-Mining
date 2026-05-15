from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .targets import add_targets
from .utils import compute_rsi, consecutive_count, rolling_percentile, rolling_trend_stats, safe_div


def build_intraday_features(m5: pd.DataFrame) -> pd.DataFrame:
    if m5.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})

    work = m5.sort_values(["date", "time"]).copy()
    work["bar_return"] = work.groupby("date")["close"].pct_change()
    work["bar_oc_return"] = safe_div(work["close"], work["open"]) - 1
    work["bar_range"] = safe_div(work["high"] - work["low"], work["open"])
    work["bar_vwap"] = safe_div(work["amount"], work["volume"]).where(lambda s: np.isfinite(s))

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
        return _safe_corr(valid, valid.shift(1), min_samples=min_samples)

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

        close30 = grp.iloc[-6:] if len(grp) >= 6 else grp
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
    rolling_cols = [
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
    ]
    for col in rolling_cols:
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
    return intraday.replace([np.inf, -np.inf], np.nan)


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
    signed_volume_pressure = np.sign(out["ret_1"].fillna(0.0)) * out["volume_change_1"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

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
        out[f"amount_abnormality_{w}"] = (out["amount"] - amt_ma) / out["amount"].rolling(
            w, min_periods=max(3, w // 2)
        ).std()
        out[f"volume_abnormality_{w}"] = (out["volume"] - vol_ma) / out["volume"].rolling(
            w, min_periods=max(3, w // 2)
        ).std()
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
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
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
