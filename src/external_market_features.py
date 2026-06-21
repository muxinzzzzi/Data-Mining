from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


INDEX_PREFIXES = ("hs300", "zz500", "zz1000", "sz50", "cyb", "sse")

RAW_USED_FILES = {
    "index": "index_daily.csv",
    "margin": "margin_daily.csv",
    "shibor": "shibor_daily.csv",
}

EXTERNAL_SOURCE_SCORE_COLUMNS = [
    "external_index_risk_score",
    "external_margin_risk_score",
    "external_shibor_risk_score",
    "external_market_risk_score",
]

EXTERNAL_STRATEGY_CONTEXT_COLUMNS = [
    "external_index_risk_score",
    "external_margin_risk_score",
    "external_shibor_risk_score",
    "external_market_risk_score",
    "external_risk_low",
    "external_risk_mid",
    "external_risk_high",
    "external_style_support",
    "external_liquidity_support",
    "external_weight_index_only",
    "external_weight_index_margin",
    "external_weight_index_shibor",
    "external_weight_all",
    "external_condition_index_trend_weak",
    "external_condition_index_drawdown",
    "external_condition_index_volatility",
    "external_condition_index_liquidity",
    "external_condition_style_weak",
    "external_condition_margin_risk",
    "external_condition_shibor_tightening",
    "external_small_vs_large_ret_20",
    "external_mid_vs_large_ret_20",
    "external_growth_vs_large_ret_20",
    "margin_balance_chg_5",
    "margin_balance_chg_20",
    "margin_buy_chg_5",
    "margin_buy_chg_20",
    "margin_balance_ma_gap_20",
    "margin_balance_z_60",
    "margin_buy_z_20",
    "margin_risk_off",
    "shibor_on_chg_5",
    "shibor_1w_chg_5",
    "shibor_1m_chg_20",
    "shibor_1w_z_60",
    "shibor_1m_z_60",
    "shibor_slope_1m_on",
    "shibor_slope_1m_1w",
    "shibor_tightening",
]


def parse_trade_date(series: pd.Series) -> pd.Series:
    clean = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return pd.to_datetime(clean, format="%Y%m%d", errors="coerce")


def _read_raw_csv(raw_dir: Path, filename: str) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
    df = pd.read_csv(path, low_memory=False)
    if "trade_date" not in df.columns:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
    df = df.copy()
    df["date"] = parse_trade_date(df["trade_date"])
    return df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods, fill_method=None).replace([np.inf, -np.inf], np.nan)


def _rolling_z(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(5, window // 2)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan)


def _rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    running_high = close.rolling(window, min_periods=max(5, window // 2)).max()
    return (close / running_high - 1).replace([np.inf, -np.inf], np.nan)


def _bool_float(condition: pd.Series) -> pd.Series:
    return condition.fillna(False).astype(float)


def _safe_sum(columns: Iterable[pd.Series], index: pd.Index) -> pd.Series:
    total = pd.Series(0.0, index=index)
    for col in columns:
        total = total.add(col.fillna(0.0), fill_value=0.0)
    return total


def _external_weight_from_score(score: pd.Series, liquidity_support: pd.Series | None = None) -> pd.Series:
    clean_score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weight = pd.Series(1.0, index=score.index, dtype=float)
    weight = weight.mask(clean_score == 2, 0.95)
    weight = weight.mask(clean_score == 3, 0.90)
    weight = weight.mask(clean_score >= 4, 0.80)
    if liquidity_support is not None:
        tight_liquidity = ~liquidity_support.fillna(True).astype(bool)
        weight = weight.mask((clean_score >= 3) & tight_liquidity, np.minimum(weight, 0.85))
    return weight.clip(lower=0.0, upper=1.0)


def _build_index_features(raw_dir: Path) -> pd.DataFrame:
    df = _read_raw_csv(raw_dir, RAW_USED_FILES["index"])
    out = pd.DataFrame({"date": df["date"]})
    if df.empty:
        return out

    for prefix in INDEX_PREFIXES:
        close = _num(df, f"{prefix}_close")
        amount = _num(df, f"{prefix}_amount")
        ret_1 = _pct_change(close, 1)
        for horizon in [5, 10, 20, 60]:
            out[f"external_{prefix}_ret_{horizon}"] = _pct_change(close, horizon)
        for window in [20, 60]:
            ma = close.rolling(window, min_periods=max(5, window // 2)).mean()
            out[f"external_{prefix}_ma_gap_{window}"] = (close / ma - 1).replace([np.inf, -np.inf], np.nan)
            out[f"external_{prefix}_vol_{window}"] = ret_1.rolling(
                window,
                min_periods=max(5, window // 2),
            ).std() * math.sqrt(252)
            out[f"external_{prefix}_drawdown_{window}"] = _rolling_drawdown(close, window)
        for horizon in [5, 20]:
            out[f"external_{prefix}_amount_chg_{horizon}"] = _pct_change(amount, horizon)
        out[f"external_{prefix}_amount_z_20"] = _rolling_z(amount, 20, 10)
        out[f"external_{prefix}_vol_20_z_252"] = _rolling_z(out[f"external_{prefix}_vol_20"], 252, 80)

    out["external_small_vs_large_ret_20"] = out["external_zz1000_ret_20"] - out["external_hs300_ret_20"]
    out["external_mid_vs_large_ret_20"] = out["external_zz500_ret_20"] - out["external_hs300_ret_20"]
    out["external_growth_vs_large_ret_20"] = out["external_cyb_ret_20"] - out["external_hs300_ret_20"]
    out["external_small_vs_large_ret_60"] = out["external_zz1000_ret_60"] - out["external_hs300_ret_60"]
    out["external_mid_vs_large_ret_60"] = out["external_zz500_ret_60"] - out["external_hs300_ret_60"]
    out["external_growth_vs_large_ret_60"] = out["external_cyb_ret_60"] - out["external_hs300_ret_60"]

    trend_weak = (out["external_zz1000_ret_20"] < 0) | (out["external_zz1000_ma_gap_60"] < 0)
    index_drawdown = (out["external_zz1000_drawdown_20"] < -0.06) | (out["external_zz1000_drawdown_60"] < -0.10)
    index_volatility = (
        out["external_zz1000_vol_20"] > 1.15 * out["external_zz1000_vol_60"]
    ) | (out["external_zz1000_vol_20_z_252"] > 1.0)
    index_liquidity = (out["external_zz1000_amount_z_20"] < -1.0) | (out["external_zz1000_amount_chg_20"] < -0.20)
    style_weak = (
        (out["external_small_vs_large_ret_20"] < -0.02)
        & (out["external_zz1000_ret_20"] < out["external_hs300_ret_20"])
    )

    out["external_condition_index_trend_weak"] = _bool_float(trend_weak)
    out["external_condition_index_drawdown"] = _bool_float(index_drawdown)
    out["external_condition_index_volatility"] = _bool_float(index_volatility)
    out["external_condition_index_liquidity"] = _bool_float(index_liquidity)
    out["external_condition_style_weak"] = _bool_float(style_weak)
    out["external_index_risk_score"] = _safe_sum(
        [
            out["external_condition_index_trend_weak"],
            out["external_condition_index_drawdown"],
            out["external_condition_index_volatility"],
            out["external_condition_index_liquidity"],
            out["external_condition_style_weak"],
        ],
        out.index,
    )
    out["external_style_support"] = _bool_float(
        ((out["external_small_vs_large_ret_20"] > 0) & (out["external_zz1000_ret_20"] > 0))
        | ((out["external_small_vs_large_ret_60"] > 0) & (out["external_zz1000_ma_gap_20"] > 0))
    )
    return out


def _build_margin_features(raw_dir: Path) -> pd.DataFrame:
    df = _read_raw_csv(raw_dir, RAW_USED_FILES["margin"])
    out = pd.DataFrame({"date": df["date"]})
    if df.empty:
        out["external_margin_risk_score"] = 0.0
        return out

    balance = _num(df, "sse_margin_balance")
    buy = _num(df, "sse_margin_buy")
    total = _num(df, "sse_margin_total")

    out["margin_balance_chg_5"] = _pct_change(balance, 5)
    out["margin_balance_chg_20"] = _pct_change(balance, 20)
    out["margin_total_chg_5"] = _pct_change(total, 5)
    out["margin_total_chg_20"] = _pct_change(total, 20)
    out["margin_buy_chg_5"] = _pct_change(buy, 5)
    out["margin_buy_chg_20"] = _pct_change(buy, 20)
    out["margin_balance_ma_gap_20"] = (balance / balance.rolling(20, min_periods=10).mean() - 1).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    out["margin_balance_z_60"] = _rolling_z(balance, 60, 20)
    out["margin_buy_z_20"] = _rolling_z(buy, 20, 10)
    margin_risk = (
        (out["margin_balance_chg_20"] < -0.03)
        | ((out["margin_buy_z_20"] < -1.0) & (out["margin_balance_ma_gap_20"] < 0))
        | ((out["margin_total_chg_20"] < -0.03) & (out["margin_buy_chg_5"] < -0.10))
    )
    out["margin_risk_off"] = _bool_float(margin_risk)
    out["external_condition_margin_risk"] = out["margin_risk_off"]
    out["external_margin_risk_score"] = out["external_condition_margin_risk"]
    return out


def _build_shibor_features(raw_dir: Path) -> pd.DataFrame:
    df = _read_raw_csv(raw_dir, RAW_USED_FILES["shibor"])
    out = pd.DataFrame({"date": df["date"]})
    if df.empty:
        out["external_shibor_risk_score"] = 0.0
        return out

    shibor_on = _num(df, "shibor_on")
    shibor_1w = _num(df, "shibor_1w")
    shibor_1m = _num(df, "shibor_1m")
    for name, series in [("on", shibor_on), ("1w", shibor_1w), ("1m", shibor_1m)]:
        out[f"shibor_{name}_chg_5"] = series.diff(5)
        out[f"shibor_{name}_chg_20"] = series.diff(20)
        out[f"shibor_{name}_z_60"] = _rolling_z(series, 60, 20)
    out["shibor_slope_1m_on"] = shibor_1m - shibor_on
    out["shibor_slope_1m_1w"] = shibor_1m - shibor_1w
    out["shibor_slope_1w_on"] = shibor_1w - shibor_on
    tightening = (
        (out["shibor_1w_chg_5"] > 0.15)
        | (out["shibor_1m_chg_20"] > 0.30)
        | ((out["shibor_1w_z_60"] > 1.0) & (out["shibor_slope_1m_on"] > 0))
    )
    out["shibor_tightening"] = _bool_float(tightening)
    out["external_condition_shibor_tightening"] = out["shibor_tightening"]
    out["external_shibor_risk_score"] = out["external_condition_shibor_tightening"]
    return out


def _combine_sources(index_features: pd.DataFrame, margin_features: pd.DataFrame, shibor_features: pd.DataFrame) -> pd.DataFrame:
    dates = pd.concat(
        [index_features[["date"]], margin_features[["date"]], shibor_features[["date"]]],
        ignore_index=True,
    ).dropna().drop_duplicates().sort_values("date")
    out = dates.reset_index(drop=True)
    for feature_df in [index_features, margin_features, shibor_features]:
        if feature_df.empty:
            continue
        out = out.merge(feature_df, on="date", how="left")

    score_defaults = {
        "external_index_risk_score": 0.0,
        "external_margin_risk_score": 0.0,
        "external_shibor_risk_score": 0.0,
    }
    for col, default in score_defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(default)

    condition_cols = [
        "external_condition_index_trend_weak",
        "external_condition_index_drawdown",
        "external_condition_index_volatility",
        "external_condition_index_liquidity",
        "external_condition_style_weak",
        "external_condition_margin_risk",
        "external_condition_shibor_tightening",
    ]
    for col in condition_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)

    out = out.copy()
    out["external_market_risk_score"] = _safe_sum([out[col] for col in condition_cols], out.index)
    if "external_style_support" not in out.columns:
        out["external_style_support"] = 0.0
    out["external_style_support"] = out["external_style_support"].fillna(0.0)
    liquidity_support = ~(
        (out["external_condition_index_liquidity"].fillna(0.0) > 0)
        | (out["external_condition_margin_risk"].fillna(0.0) > 0)
    )
    out["external_liquidity_support"] = liquidity_support.astype(float)
    out["external_risk_low"] = (out["external_market_risk_score"] <= 1).astype(float)
    out["external_risk_mid"] = (
        (out["external_market_risk_score"] >= 2) & (out["external_market_risk_score"] <= 3)
    ).astype(float)
    out["external_risk_high"] = (out["external_market_risk_score"] >= 4).astype(float)

    out["external_weight_index_only"] = _external_weight_from_score(
        out["external_index_risk_score"],
        out["external_liquidity_support"].astype(bool),
    )
    out["external_weight_index_margin"] = _external_weight_from_score(
        out["external_index_risk_score"] + out["external_margin_risk_score"],
        out["external_liquidity_support"].astype(bool),
    )
    out["external_weight_index_shibor"] = _external_weight_from_score(
        out["external_index_risk_score"] + out["external_shibor_risk_score"],
        out["external_liquidity_support"].astype(bool),
    )
    out["external_weight_all"] = _external_weight_from_score(
        out["external_market_risk_score"],
        out["external_liquidity_support"].astype(bool),
    )
    return out.replace([np.inf, -np.inf], np.nan)


def _neutral_external_frame(target_dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.to_datetime(target_dates).dropna().sort_values().drop_duplicates()})
    for col in EXTERNAL_STRATEGY_CONTEXT_COLUMNS:
        out[col] = 0.0
    out["external_liquidity_support"] = 1.0
    out["external_weight_index_only"] = 1.0
    out["external_weight_index_margin"] = 1.0
    out["external_weight_index_shibor"] = 1.0
    out["external_weight_all"] = 1.0
    return out.reset_index(drop=True)


def build_external_market_features(raw_dir: str | Path, target_dates: pd.Series | pd.DatetimeIndex) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    target = pd.Series(pd.to_datetime(target_dates, errors="coerce"), name="date").dropna()
    if target.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})

    index_features = _build_index_features(raw_path)
    margin_features = _build_margin_features(raw_path)
    shibor_features = _build_shibor_features(raw_path)
    combined = _combine_sources(index_features, margin_features, shibor_features)
    if combined.empty:
        return _neutral_external_frame(target)

    value_cols = [col for col in combined.columns if col != "date"]
    shifted = combined.sort_values("date").copy()
    shifted[value_cols] = shifted[value_cols].shift(1)
    target_df = pd.DataFrame({"date": target.sort_values().drop_duplicates().reset_index(drop=True)})
    aligned = pd.merge_asof(
        target_df,
        shifted.sort_values("date"),
        on="date",
        direction="backward",
    )

    score_cols = [col for col in aligned.columns if col.endswith("_risk_score")] + [
        "external_market_risk_score",
        "external_risk_low",
        "external_risk_mid",
        "external_risk_high",
        "external_style_support",
        "external_condition_index_trend_weak",
        "external_condition_index_drawdown",
        "external_condition_index_volatility",
        "external_condition_index_liquidity",
        "external_condition_style_weak",
        "external_condition_margin_risk",
        "external_condition_shibor_tightening",
        "margin_risk_off",
        "shibor_tightening",
    ]
    for col in score_cols:
        if col in aligned.columns:
            aligned[col] = aligned[col].fillna(0.0)
    if "external_market_risk_score" in aligned.columns:
        market_score = aligned["external_market_risk_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        aligned["external_risk_low"] = (market_score <= 1).astype(float)
        aligned["external_risk_mid"] = ((market_score >= 2) & (market_score <= 3)).astype(float)
        aligned["external_risk_high"] = (market_score >= 4).astype(float)
    for col in [
        "external_weight_index_only",
        "external_weight_index_margin",
        "external_weight_index_shibor",
        "external_weight_all",
    ]:
        if col in aligned.columns:
            aligned[col] = aligned[col].fillna(1.0)
    if "external_liquidity_support" in aligned.columns:
        aligned["external_liquidity_support"] = aligned["external_liquidity_support"].fillna(1.0)
    return aligned.replace([np.inf, -np.inf], np.nan)


def external_feature_columns(columns: Iterable[str]) -> list[str]:
    return [
        col
        for col in columns
        if col.startswith("external_") or col.startswith("margin_") or col.startswith("shibor_")
    ]
