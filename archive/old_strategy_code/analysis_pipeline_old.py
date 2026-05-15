from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

DAILY_PATH = ROOT / "SH#880823_daily_clean.csv"
M5_PATH = ROOT / "SH#880823_5min_clean.csv"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-05-06")
INITIAL_CAPITAL = 100_000.0
COST_RATE = 0.001
RETRAIN_EVERY = 20
HORIZON = 5
REBALANCE_FREQ = 5


def prepare_matplotlib() -> None:
    cache_dir = ROOT / ".mplconfig"
    cache_dir.mkdir(exist_ok=True)
    (ROOT / ".cache").mkdir(exist_ok=True)
    matplotlib.use("Agg")


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.divide(b.replace(0, np.nan))


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    if M5_PATH.exists():
        m5 = pd.read_csv(M5_PATH)
        if not m5.empty:
            m5["time"] = m5["time"].astype(str).str.zfill(5)
            m5["datetime"] = pd.to_datetime(m5["date"] + " " + m5["time"])
            m5["date"] = pd.to_datetime(m5["date"])
        else:
            m5 = pd.DataFrame(columns=["date", "time", "datetime"])
    else:
        m5 = pd.DataFrame(columns=["date", "time", "datetime"])
    return daily, m5


def build_intraday_features(m5: pd.DataFrame) -> pd.DataFrame:
    if m5.empty:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})

    work = m5.sort_values(["date", "time"]).copy()
    work["bar_return"] = safe_div(work["close"], work["open"]) - 1
    work["bar_range"] = safe_div(work["high"] - work["low"], work["open"])
    work["vwap_bar"] = safe_div(work["amount"], work["volume"])

    rows = []
    for date, grp in work.groupby("date", sort=True):
        grp = grp.reset_index(drop=True)
        first_open = grp.loc[0, "open"]
        last_close = grp.loc[len(grp) - 1, "close"]
        morning_slice = grp[grp["time"] <= "11:30"]
        afternoon_slice = grp[grp["time"] >= "13:05"]
        first_hour_slice = grp[grp["time"] <= "10:30"]
        last_hour_slice = grp[grp["time"] >= "14:00"]
        midday_slice = grp[grp["time"] <= "14:00"]

        morning_close = morning_slice.iloc[-1]["close"] if not morning_slice.empty else last_close
        afternoon_open = afternoon_slice.iloc[0]["open"] if not afternoon_slice.empty else grp.iloc[-1]["open"]
        first_hour_close = first_hour_slice.iloc[-1]["close"] if not first_hour_slice.empty else morning_close
        last_hour_open = last_hour_slice.iloc[0]["open"] if not last_hour_slice.empty else grp.iloc[-1]["open"]
        midday_close = midday_slice.iloc[-1]["close"] if not midday_slice.empty else last_close

        morning_return = morning_close / first_open - 1
        afternoon_return = last_close / afternoon_open - 1
        first_hour_return = first_hour_close / first_open - 1
        last_hour_return = last_close / last_hour_open - 1
        intraday_return = last_close / first_open - 1
        intraday_range = grp["high"].max() / grp["low"].min() - 1
        midday_reversal = last_close / midday_close - 1
        total_volume = grp["volume"].sum()
        volume_profile = grp["volume"].iloc[:12].sum() / total_volume if total_volume else np.nan

        rows.append(
            {
                "date": date,
                "m5_intraday_return": intraday_return,
                "m5_morning_return": morning_return,
                "m5_afternoon_return": afternoon_return,
                "m5_first_hour_return": first_hour_return,
                "m5_last_hour_return": last_hour_return,
                "m5_midday_reversal": midday_reversal,
                "m5_intraday_range": intraday_range,
                "m5_bar_volatility": grp["bar_return"].std(),
                "m5_bar_range_mean": grp["bar_range"].mean(),
                "m5_bar_range_max": grp["bar_range"].max(),
                "m5_volume_profile_am": volume_profile,
                "m5_total_volume": grp["volume"].sum(),
                "m5_total_amount": grp["amount"].sum(),
                "m5_avg_vwap": grp["vwap_bar"].mean(),
            }
        )

    intraday = pd.DataFrame(rows).sort_values("date")
    intraday["m5_overnight_gap"] = intraday["m5_intraday_return"].shift(1)
    intraday["m5_morning_mean_5"] = intraday["m5_morning_return"].rolling(5).mean()
    intraday["m5_afternoon_mean_5"] = intraday["m5_afternoon_return"].rolling(5).mean()
    intraday["m5_range_mean_5"] = intraday["m5_intraday_range"].rolling(5).mean()
    intraday["m5_reversal_mean_5"] = intraday["m5_midday_reversal"].rolling(5).mean()
    return intraday


def build_daily_features(daily: pd.DataFrame, intraday: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("date").copy()
    df = df.merge(intraday, on="date", how="left")
    m5_cols = [col for col in df.columns if col.startswith("m5_")]
    if m5_cols:
        df["m5_available"] = df["m5_intraday_return"].notna().astype(float)
        df[m5_cols] = df[m5_cols].fillna(0.0)
    else:
        df["m5_available"] = 0.0

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_10"] = df["close"].pct_change(10)
    df["ret_20"] = df["close"].pct_change(20)
    df["ret_60"] = df["close"].pct_change(60)
    df["gap_open"] = safe_div(df["open"], df["close"].shift(1)) - 1
    df["close_to_high"] = safe_div(df["close"], df["high"]) - 1
    df["close_to_low"] = safe_div(df["close"], df["low"]) - 1
    df["intraday_body"] = safe_div(df["close"] - df["open"], df["open"])
    df["daily_range_pct"] = safe_div(df["high"] - df["low"], df["close"].shift(1))

    for window in [5, 10, 20, 60, 120]:
        ma = df["close"].rolling(window).mean()
        df[f"ma_ratio_{window}"] = safe_div(df["close"], ma) - 1
        df[f"vol_ratio_{window}"] = safe_div(df["volume"], df["volume"].rolling(window).mean()) - 1
        df[f"amount_ratio_{window}"] = safe_div(df["amount"], df["amount"].rolling(window).mean()) - 1
        df[f"volatility_{window}"] = df["ret_1"].rolling(window).std()

    df["ma_spread_10_20"] = df["close"].rolling(10).mean() / df["close"].rolling(20).mean() - 1
    df["ma_spread_20_60"] = df["close"].rolling(20).mean() / df["close"].rolling(60).mean() - 1
    df["rolling_high_20"] = df["high"].rolling(20).max()
    df["rolling_low_10"] = df["low"].rolling(10).min()
    df["breakout_20"] = df["close"] / df["rolling_high_20"].shift(1) - 1
    df["breakdown_10"] = df["close"] / df["rolling_low_10"].shift(1) - 1
    df["rsi_14"] = compute_rsi(df["close"], 14)
    df["drawdown_60"] = df["close"] / df["close"].rolling(60).max() - 1
    df["drawdown_120"] = df["close"] / df["close"].rolling(120).max() - 1
    df["close_rank_20"] = (
        df["close"].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    )
    df["ret_mean_5"] = df["ret_1"].rolling(5).mean()
    df["ret_mean_20"] = df["ret_1"].rolling(20).mean()
    df["ret_skew_20"] = df["ret_1"].rolling(20).skew()
    df["volatility_ratio_20_60"] = safe_div(df["volatility_20"], df["volatility_60"])
    df["range_mean_5"] = df["daily_range_pct"].rolling(5).mean()
    df["range_mean_20"] = df["daily_range_pct"].rolling(20).mean()
    df["rsi_change_5"] = df["rsi_14"].diff(5)
    df["trend_strength"] = 0.4 * df["ma_ratio_20"] + 0.4 * df["ma_ratio_60"] + 0.2 * df["ret_20"]

    if "m5_intraday_return" in df.columns:
        df["m5_consistency"] = df["m5_intraday_return"] - df["intraday_body"]
        df["m5_vwap_discount"] = safe_div(df["close"], df["m5_avg_vwap"]) - 1
        df["m5_amount_ratio_gap"] = safe_div(df["amount"], df["m5_total_amount"]) - 1
    else:
        df["m5_consistency"] = np.nan
        df["m5_vwap_discount"] = np.nan
        df["m5_amount_ratio_gap"] = np.nan

    df["fwd_ret_1d"] = df["close"].shift(-1) / df["close"] - 1
    df["target_ret_5d"] = df["close"].shift(-HORIZON) / df["close"] - 1
    df["target_up_5d"] = (df["target_ret_5d"] > 0).astype(float)
    df["trade_date"] = df["date"].shift(-1)
    df["next_date"] = df["date"].shift(-HORIZON)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "symbol",
        "name",
        "frequency",
        "adjustment",
        "date",
        "time",
        "trade_date",
        "next_date",
        "fwd_ret_1d",
        "target_ret_5d",
        "target_up_5d",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "m5_total_volume",
        "m5_total_amount",
        "m5_avg_vwap",
        "rolling_high_20",
        "rolling_low_10",
    }
    feature_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def clip_by_train_quantiles(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for col in feature_cols:
        lo = train[col].quantile(lower)
        hi = train[col].quantile(upper)
        train[col] = train[col].clip(lo, hi)
        test[col] = test[col].clip(lo, hi)
    return train, test


def walk_forward_predictions(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    feature_ready = df.dropna(subset=feature_cols + ["fwd_ret_1d"]).copy()
    train_ready = feature_ready.dropna(subset=["target_ret_5d", "target_up_5d"]).copy()
    test_dates = (
        feature_ready.loc[
            (feature_ready["date"] >= TEST_START) & (feature_ready["date"] <= TEST_END),
            "date",
        ]
        .sort_values()
        .tolist()
    )
    preds = []

    reg_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=3,
        max_iter=250,
        min_samples_leaf=20,
        l2_regularization=0.1,
    )
    cls_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=3,
        max_iter=250,
        min_samples_leaf=20,
        l2_regularization=0.1,
    )

    for idx in range(0, len(test_dates), RETRAIN_EVERY):
        chunk_dates = test_dates[idx : idx + RETRAIN_EVERY]
        if not chunk_dates:
            continue

        train_mask = train_ready["date"] < chunk_dates[0]
        test_mask = feature_ready["date"].isin(chunk_dates)
        train = train_ready.loc[train_mask].copy()
        test = feature_ready.loc[test_mask].copy()
        if len(train) < 252 or test.empty:
            continue

        train, test = clip_by_train_quantiles(train, test, feature_cols)

        x_train = train[feature_cols]
        y_reg = train["target_ret_5d"]
        y_cls = train["target_up_5d"].astype(int)

        reg_model.fit(x_train, y_reg)
        cls_model.fit(x_train, y_cls)

        test = test.copy()
        test["pred_ret_5d"] = reg_model.predict(test[feature_cols])
        test["pred_up_prob_5d"] = cls_model.predict_proba(test[feature_cols])[:, 1]
        test["pred_up_5d"] = (test["pred_up_prob_5d"] >= 0.5).astype(int)
        preds.append(test)

    if not preds:
        raise RuntimeError("No out-of-sample predictions were generated.")

    pred_df = pd.concat(preds, ignore_index=True).sort_values("date")
    return pred_df


def regression_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    eval_df = pred_df.dropna(subset=["target_ret_5d"]).copy()
    y_true = eval_df["target_ret_5d"]
    y_pred = eval_df["pred_ret_5d"]
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(eval_df) > 2 else np.nan
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "directional_accuracy": ((y_true > 0) == (y_pred > 0)).mean(),
        "correlation": corr,
    }


def classification_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    eval_df = pred_df.dropna(subset=["target_up_5d"]).copy()
    y_true = eval_df["target_up_5d"].astype(int)
    y_pred = eval_df["pred_up_5d"].astype(int)
    y_prob = eval_df["pred_up_prob_5d"]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_true.nunique() > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc"] = np.nan
    return metrics


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / 252
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1 / years) - 1


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1


def rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )


def strategy_metrics(strategy_df: pd.DataFrame, strategy_col: str, benchmark_col: str = "buy_hold_ret") -> dict[str, float]:
    ret = strategy_df[strategy_col]
    benchmark = strategy_df[benchmark_col]
    equity = (1 + ret).cumprod() * INITIAL_CAPITAL
    bench_equity = (1 + benchmark).cumprod() * INITIAL_CAPITAL
    active = ret - benchmark
    vol = ret.std() * np.sqrt(252)
    ann_ret = annualized_return(equity)
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan
    mdd = max_drawdown(equity)
    tracking_error = active.std() * np.sqrt(252)
    annualized_excess = active.mean() * 252
    return {
        "total_return": equity.iloc[-1] / INITIAL_CAPITAL - 1,
        "annual_return": ann_ret,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": ann_ret / abs(mdd) if mdd < 0 and pd.notna(ann_ret) else np.nan,
        "win_rate": (ret > 0).mean(),
        "avg_turnover": strategy_df[strategy_col.replace("_ret", "_turnover")].mean(),
        "benchmark_total_return": bench_equity.iloc[-1] / INITIAL_CAPITAL - 1,
        "excess_return_vs_buy_hold": (equity.iloc[-1] / INITIAL_CAPITAL) - (bench_equity.iloc[-1] / INITIAL_CAPITAL),
        "annualized_excess_return": annualized_excess,
        "tracking_error": tracking_error,
        "information_ratio": annualized_excess / tracking_error if tracking_error > 0 else np.nan,
    }


@dataclass
class SignalPack:
    positions: pd.Series
    turnover: pd.Series
    returns: pd.Series


def apply_rebalance_schedule(raw_positions: pd.Series, rebalance_freq: int = REBALANCE_FREQ) -> pd.Series:
    raw_positions = raw_positions.ffill().fillna(0.0).clip(lower=0.0, upper=1.0)
    effective = []
    current = raw_positions.iloc[0] if len(raw_positions) else 0.0
    for idx, value in enumerate(raw_positions):
        if idx == 0 or idx % rebalance_freq == 0:
            current = value
        effective.append(current)
    return pd.Series(effective, index=raw_positions.index)


def positions_to_returns(df: pd.DataFrame, positions: pd.Series) -> SignalPack:
    effective_positions = apply_rebalance_schedule(positions)
    turnover = effective_positions.diff().abs().fillna(effective_positions.abs())
    realized = effective_positions * df["fwd_ret_1d"].fillna(0) - turnover * COST_RATE
    return SignalPack(positions=effective_positions, turnover=turnover, returns=realized)


def build_strategy_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_df.sort_values("date").copy()
    df["pred_ret_rank_60"] = rolling_percentile(df["pred_ret_5d"], 60, 20)
    df["pred_prob_rank_60"] = rolling_percentile(df["pred_up_prob_5d"], 60, 20)
    df["pred_ret_rank_20"] = rolling_percentile(df["pred_ret_5d"], 20, 10)
    df["pred_prob_rank_20"] = rolling_percentile(df["pred_up_prob_5d"], 20, 10)

    df["mom_signal"] = ((df["ret_20"] > 0) & (df["ma_ratio_20"] > 0)).astype(float)
    df["ma_signal"] = ((df["ma_spread_10_20"] > 0) & (df["ma_spread_20_60"] > -0.01)).astype(float)
    df["breakout_signal"] = ((df["breakout_20"] > 0) & (df["drawdown_60"] > -0.12)).astype(float)
    df["intraday_reversal_signal"] = (
        (df["m5_morning_return"] < -0.004)
        & (df["m5_afternoon_return"] > 0.002)
        & (df["m5_intraday_range"] > df["m5_range_mean_5"])
    ).astype(float)

    tech_score = (
        0.30 * df["mom_signal"]
        + 0.25 * df["ma_signal"]
        + 0.25 * df["breakout_signal"]
        + 0.20 * df["intraday_reversal_signal"]
    ).fillna(0.0)
    model_score = (
        0.35 * df["pred_ret_rank_60"].fillna(0.5)
        + 0.30 * df["pred_prob_rank_60"].fillna(0.5)
        + 0.20 * (df["pred_ret_5d"] > 0).astype(float)
        + 0.15 * (df["pred_up_prob_5d"] > 0.55).astype(float)
    )

    raw_ml_pos = 0.62 + 0.18 * model_score + 0.10 * tech_score + 0.10 * df["pred_ret_rank_20"].fillna(0.5)
    raw_ml_pos = raw_ml_pos.clip(0.45, 1.00)

    below_ma60 = df["ma_ratio_60"] < 0
    neg_mom_20 = df["ret_20"] < 0
    high_vol = df["volatility_20"] > df["volatility_60"].fillna(df["volatility_20"]) * 1.20
    deep_drawdown = df["drawdown_60"] < -0.12
    very_deep_drawdown = df["drawdown_120"] < -0.18
    weak_model = (df["pred_ret_5d"] < 0) & (df["pred_up_prob_5d"] < 0.50)

    risk_multiplier = pd.Series(1.0, index=df.index)
    risk_multiplier = risk_multiplier.where(~below_ma60, 0.90)
    risk_multiplier = risk_multiplier.where(~neg_mom_20, risk_multiplier * 0.90)
    risk_multiplier = risk_multiplier.where(~high_vol, risk_multiplier * 0.85)
    risk_multiplier = risk_multiplier.where(~deep_drawdown, risk_multiplier * 0.80)
    risk_multiplier = risk_multiplier.where(~very_deep_drawdown, risk_multiplier * 0.75)
    risk_multiplier = risk_multiplier.where(~weak_model, risk_multiplier * 0.90)

    df["buy_hold_pos"] = 1.0
    df["momentum_pos"] = df["mom_signal"]
    df["ma_pos"] = df["ma_signal"]
    df["breakout_pos"] = df["breakout_signal"]
    df["intraday_reversal_pos"] = df["intraday_reversal_signal"]
    df["ml_enhanced_raw_pos"] = (raw_ml_pos * risk_multiplier).clip(0.35, 1.00)
    df.loc[(df["pred_prob_rank_60"] > 0.8) & (df["trend_strength"] > 0), "ml_enhanced_raw_pos"] = (
        df.loc[(df["pred_prob_rank_60"] > 0.8) & (df["trend_strength"] > 0), "ml_enhanced_raw_pos"] + 0.10
    ).clip(upper=1.0)
    df.loc[(df["pred_prob_rank_60"] < 0.2) & (df["ret_20"] < 0), "ml_enhanced_raw_pos"] = (
        df.loc[(df["pred_prob_rank_60"] < 0.2) & (df["ret_20"] < 0), "ml_enhanced_raw_pos"] * 0.5
    ).clip(lower=0.30)

    df["ml_enhanced_pos"] = df["ml_enhanced_raw_pos"]

    for name in ["buy_hold", "momentum", "ma", "breakout", "intraday_reversal", "ml_enhanced"]:
        pack = positions_to_returns(df, df[f"{name}_pos"])
        df[f"{name}_pos"] = pack.positions
        df[f"{name}_turnover"] = pack.turnover
        df[f"{name}_ret"] = pack.returns
        df[f"{name}_equity"] = (1 + pack.returns).cumprod() * INITIAL_CAPITAL

    df["ml_active_ret"] = df["ml_enhanced_ret"] - df["buy_hold_ret"]
    df["ml_active_equity"] = (1 + df["ml_active_ret"]).cumprod() * INITIAL_CAPITAL
    df["buy_hold_drawdown"] = drawdown_series(df["buy_hold_equity"])
    df["ml_enhanced_drawdown"] = drawdown_series(df["ml_enhanced_equity"])
    return df


def save_plots(pred_df: pd.DataFrame, strategy_df: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df["trade_date"], strategy_df["buy_hold_equity"], label="Buy & Hold")
    plt.plot(strategy_df["trade_date"], strategy_df["ml_enhanced_equity"], label="ML Enhanced")
    plt.title("Buy-and-Hold vs ML-Enhanced Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_buyhold_vs_ml.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df["trade_date"], strategy_df["ml_active_equity"], label="Active Equity")
    plt.axhline(INITIAL_CAPITAL, color="black", linewidth=0.8, linestyle="--")
    plt.title("Excess Equity Curve: ML Enhanced Relative to Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Active Equity (RMB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "excess_equity_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df["trade_date"], strategy_df["buy_hold_drawdown"], label="Buy & Hold")
    plt.plot(strategy_df["trade_date"], strategy_df["ml_enhanced_drawdown"], label="ML Enhanced")
    plt.title("Drawdown Curve: Buy-and-Hold vs ML-Enhanced")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_buyhold_vs_ml.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(pred_df["date"], pred_df["pred_up_prob_5d"], label="Predicted Up Probability (5D)")
    if pred_df["target_up_5d"].notna().any():
        plt.plot(pred_df["date"], pred_df["target_up_5d"], label="Actual Up Label (5D)", alpha=0.5)
    plt.title("Predicted 5-Day Up Probability Over Time")
    plt.xlabel("Date")
    plt.ylabel("Probability / Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "predicted_probability_over_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(strategy_df["trade_date"], strategy_df["buy_hold_pos"], label="Buy & Hold Exposure")
    plt.plot(strategy_df["trade_date"], strategy_df["ml_enhanced_pos"], label="ML Enhanced Exposure")
    plt.title("Strategy Exposure Over Time")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "position_exposure_over_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.scatter(pred_df["pred_ret_5d"], pred_df["target_ret_5d"], alpha=0.6)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Predicted vs Actual 5-Day Forward Returns")
    plt.xlabel("Predicted 5-Day Return")
    plt.ylabel("Actual 5-Day Return")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "prediction_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(
        ["BuyHold", "Momentum", "MA", "Breakout", "IntradayRev", "MLEnhanced"],
        [
            strategy_df["buy_hold_ret"].mean() * 252,
            strategy_df["momentum_ret"].mean() * 252,
            strategy_df["ma_ret"].mean() * 252,
            strategy_df["breakout_ret"].mean() * 252,
            strategy_df["intraday_reversal_ret"].mean() * 252,
            strategy_df["ml_enhanced_ret"].mean() * 252,
        ],
    )
    plt.title("Approximate Annualized Mean Return by Strategy")
    plt.ylabel("Annualized Return")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "strategy_return_bars.png", dpi=160)
    plt.close()


def export_summary(
    pred_df: pd.DataFrame,
    strategy_df: pd.DataFrame,
    reg_metrics: dict[str, float],
    cls_metrics: dict[str, float],
    strategy_table: pd.DataFrame,
) -> None:
    best_strategy = strategy_table.sort_values("sharpe", ascending=False).iloc[0]
    ml_row = strategy_table.loc[strategy_table["strategy"] == "ml_enhanced"].iloc[0]
    benchmark_row = strategy_table.loc[strategy_table["strategy"] == "buy_hold"].iloc[0]
    summary = f"""# Analysis Summary

## Setup
- Initial capital: {INITIAL_CAPITAL:,.0f} RMB
- Training window end: {TRAIN_END.date()}
- Test window: {TEST_START.date()} to {TEST_END.date()}
- This is implemented as a machine-learning-enhanced index timing strategy because only index-level data are available.
- Signal timing: generate signals after the close on date t and apply exposure to future daily returns, with a {REBALANCE_FREQ}-trading-day rebalance schedule.

## Feature Engineering
- Daily momentum, moving average spreads, rolling volatility, RSI, breakout, drawdown, range, gap, volume and amount ratios.
- 5-minute auxiliary features including morning return, afternoon return, first-hour return, last-hour return, intraday range, midday reversal, volume profile and VWAP discount.

## Predictive Models
- Regression model: HistGradientBoostingRegressor.
- Classification model: HistGradientBoostingClassifier.
- The model predicts forward {HORIZON}-day returns and forward {HORIZON}-day upward probability.
- Walk-forward retraining frequency: every {RETRAIN_EVERY} test days, without using future information.

## Prediction Metrics
- Regression MAE: {reg_metrics['mae']:.6f}
- Regression RMSE: {reg_metrics['rmse']:.6f}
- Directional accuracy: {reg_metrics['directional_accuracy']:.4f}
- Return correlation: {reg_metrics['correlation']:.4f}
- Classification accuracy: {cls_metrics['accuracy']:.4f}
- Precision: {cls_metrics['precision']:.4f}
- Recall: {cls_metrics['recall']:.4f}
- AUC: {cls_metrics['auc']:.4f}

## Enhanced Index Strategy
- The ML-enhanced strategy converts model outputs into dynamic exposure weights using predicted return, predicted probability, rolling model percentiles, technical trend confirmation, and drawdown/volatility risk filters.
- Exposure is reduced when price is below the 60-day moving average, when 20-day momentum is negative, and during high-volatility or deep-drawdown periods.
- The strategy is evaluated directly against the buy-and-hold benchmark.

## Benchmark Comparison
- Buy-and-hold total return: {benchmark_row['total_return']:.2%}
- ML-enhanced total return: {ml_row['total_return']:.2%}
- Excess return over buy-and-hold: {ml_row['excess_return_vs_buy_hold']:.2%}
- Annualized excess return: {ml_row['annualized_excess_return']:.2%}
- Tracking error: {ml_row['tracking_error']:.2%}
- Information ratio: {ml_row['information_ratio']:.4f}
- ML-enhanced Sharpe ratio: {ml_row['sharpe']:.4f}
- ML-enhanced Calmar ratio: {ml_row['calmar']:.4f}
- ML-enhanced max drawdown: {ml_row['max_drawdown']:.2%}
- ML-enhanced average turnover: {ml_row['avg_turnover']:.4f}

## Strategy Ranking
- Best Sharpe strategy: {best_strategy['strategy']}
- Best strategy total return: {best_strategy['total_return']:.2%}
- Best strategy annual return: {best_strategy['annual_return']:.2%}
- Best strategy max drawdown: {best_strategy['max_drawdown']:.2%}

## Files
- predictions: outputs/test_predictions.csv
- strategy paths: outputs/strategy_daily.csv
- model metrics: outputs/model_metrics.json
- strategy metrics: outputs/strategy_metrics.csv
- plots: outputs/plots/
"""
    (OUTPUT_DIR / "summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    prepare_matplotlib()
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    daily, m5 = load_data()
    intraday = build_intraday_features(m5)
    feature_df = build_daily_features(daily, intraday)
    feature_cols = get_feature_columns(feature_df)

    pred_df = walk_forward_predictions(feature_df, feature_cols)
    pred_df = pred_df[(pred_df["date"] >= TEST_START) & (pred_df["date"] <= TEST_END)].copy()
    pred_df = pred_df.dropna(subset=["trade_date", "fwd_ret_1d"])

    reg_metrics = regression_metrics(pred_df)
    cls_metrics = classification_metrics(pred_df)
    strategy_df = build_strategy_frame(pred_df)

    strategy_rows = []
    for name in ["buy_hold", "momentum", "ma", "breakout", "intraday_reversal", "ml_enhanced"]:
        row = strategy_metrics(strategy_df, f"{name}_ret")
        row["strategy"] = name
        strategy_rows.append(row)
    strategy_table = pd.DataFrame(strategy_rows)
    strategy_table["sharpe_rank"] = strategy_table["sharpe"].rank(ascending=False, method="dense")
    display_order = ["buy_hold", "ml_enhanced", "momentum", "ma", "breakout", "intraday_reversal"]
    strategy_table["display_order"] = strategy_table["strategy"].map({name: idx for idx, name in enumerate(display_order)})
    strategy_table = strategy_table.sort_values(["display_order", "sharpe"], ascending=[True, False]).drop(columns=["display_order"])

    pred_export_cols = [
        "date",
        "trade_date",
        "next_date",
        "close",
        "fwd_ret_1d",
        "target_ret_5d",
        "target_up_5d",
        "pred_ret_5d",
        "pred_up_prob_5d",
        "pred_up_5d",
        "pred_ret_rank_20",
        "pred_ret_rank_60",
        "pred_prob_rank_20",
        "pred_prob_rank_60",
        "ret_1",
        "ret_5",
        "ret_20",
        "ma_ratio_20",
        "ma_ratio_60",
        "volatility_20",
        "drawdown_60",
        "m5_available",
        "m5_morning_return",
        "m5_afternoon_return",
        "m5_intraday_range",
    ]
    strategy_df[pred_export_cols].to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    strategy_export_cols = [
        "date",
        "trade_date",
        "next_date",
        "fwd_ret_1d",
        "target_ret_5d",
        "pred_ret_5d",
        "pred_up_prob_5d",
        "buy_hold_pos",
        "momentum_pos",
        "ma_pos",
        "breakout_pos",
        "intraday_reversal_pos",
        "ml_enhanced_pos",
        "buy_hold_turnover",
        "ml_enhanced_turnover",
        "buy_hold_ret",
        "momentum_ret",
        "ma_ret",
        "breakout_ret",
        "intraday_reversal_ret",
        "ml_enhanced_ret",
        "ml_active_ret",
        "buy_hold_equity",
        "momentum_equity",
        "ma_equity",
        "breakout_equity",
        "intraday_reversal_equity",
        "ml_enhanced_equity",
        "ml_active_equity",
        "buy_hold_drawdown",
        "ml_enhanced_drawdown",
    ]
    strategy_df[strategy_export_cols].to_csv(OUTPUT_DIR / "strategy_daily.csv", index=False)

    ml_row = strategy_table.loc[strategy_table["strategy"] == "ml_enhanced"].iloc[0].to_dict()
    benchmark_row = strategy_table.loc[strategy_table["strategy"] == "buy_hold"].iloc[0].to_dict()
    model_metrics = {
        "regression": reg_metrics,
        "classification": cls_metrics,
        "prediction_horizon_days": HORIZON,
        "rebalance_frequency_days": REBALANCE_FREQ,
        "feature_count": len(feature_cols),
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "benchmark_comparison": {
            "buy_hold_total_return": benchmark_row["total_return"],
            "ml_enhanced_total_return": ml_row["total_return"],
            "excess_return_vs_buy_hold": ml_row["excess_return_vs_buy_hold"],
            "annualized_excess_return": ml_row["annualized_excess_return"],
            "tracking_error": ml_row["tracking_error"],
            "information_ratio": ml_row["information_ratio"],
            "ml_enhanced_max_drawdown": ml_row["max_drawdown"],
            "ml_enhanced_sharpe": ml_row["sharpe"],
            "ml_enhanced_calmar": ml_row["calmar"],
            "ml_enhanced_avg_turnover": ml_row["avg_turnover"],
        },
    }
    (OUTPUT_DIR / "model_metrics.json").write_text(
        json.dumps(model_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    strategy_table.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)

    save_plots(pred_df, strategy_df)
    export_summary(pred_df, strategy_df, reg_metrics, cls_metrics, strategy_table)

    print("Finished.")
    print(f"Features used: {len(feature_cols)}")
    print(f"Predictions exported to: {OUTPUT_DIR / 'test_predictions.csv'}")
    print(f"Strategy metrics exported to: {OUTPUT_DIR / 'strategy_metrics.csv'}")
    print(f"Best strategy: {strategy_table.iloc[0]['strategy']}")


if __name__ == "__main__":
    main()
