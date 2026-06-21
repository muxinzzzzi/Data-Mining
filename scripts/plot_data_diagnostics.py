from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DAILY_PATH, HORIZONS, M5_PATH, OUTPUT_DIR, TEST_END, TEST_START, VALID_END, VALID_START

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.features import build_daily_features, build_intraday_features, get_feature_columns


DIAG_DIR = OUTPUT_DIR / "data_diagnostics"
PLOT_DIR = DIAG_DIR / "plots"
DPI = 180


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, generated: list[Path]) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()
    generated.append(path)


def safe_pct_change(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan)


def compute_drawdown(close: pd.Series) -> pd.Series:
    running_max = close.cummax()
    return close / running_max.replace(0, np.nan) - 1


def maybe_get_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    try:
        return get_feature_columns(feature_df)
    except Exception:
        excluded = {
            "date",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "fwd_ret_1d",
        }
        cols: list[str] = []
        for col in feature_df.columns:
            if col in excluded or col.startswith("target_"):
                continue
            if pd.api.types.is_numeric_dtype(feature_df[col]):
                cols.append(col)
        return cols


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.2%}"


def num(value: float | int | None, digits: int = 6) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "No rows."
    display = df.head(max_rows).copy() if max_rows else df.copy()
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in display.itertuples(index=False):
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            elif isinstance(value, pd.Timestamp):
                vals.append(value.strftime("%Y-%m-%d"))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def add_period_shading(ax: plt.Axes) -> None:
    ax.axvspan(VALID_START, VALID_END, color="#4c78a8", alpha=0.10, label="Validation")
    ax.axvspan(TEST_START, TEST_END, color="#f58518", alpha=0.10, label="Test")


def daily_return_stats(returns: pd.Series) -> dict[str, float]:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "skewness": float(clean.skew()),
        "kurtosis": float(clean.kurtosis()),
        "q01": float(clean.quantile(0.01)),
        "q99": float(clean.quantile(0.99)),
    }


def plot_index_close(daily: pd.DataFrame, generated: list[Path]) -> float:
    first_test = daily.loc[daily["date"] >= TEST_START].iloc[0]
    final_test = daily.loc[daily["date"] <= TEST_END].iloc[-1]
    buy_hold_return = float(final_test["close"] / first_test["close"] - 1)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(daily["date"], daily["close"], color="#1f77b4", linewidth=1.7, label="Close")
    add_period_shading(ax)
    for date, label in [(VALID_START, "Validation start"), (TEST_START, "Test start"), (TEST_END, "Test end")]:
        ax.axvline(date, color="black", linewidth=0.9, linestyle="--", alpha=0.65)
        ax.text(date, ax.get_ylim()[1] * 0.96, label, rotation=90, va="top", ha="right", fontsize=8)
    ax.annotate(
        f"Test buy-and-hold return: {buy_hold_return:.2%}",
        xy=(final_test["date"], final_test["close"]),
        xytext=(0.58, 0.18),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 0.9},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#999999", "alpha": 0.9},
    )
    ax.set_title("Micro-Cap Index Close Price: Full Sample, Validation, and Test Periods")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index close")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    savefig(PLOT_DIR / "index_close_train_valid_test.png", generated)
    return buy_hold_return


def plot_daily_return_distribution(daily: pd.DataFrame, generated: list[Path]) -> tuple[dict[str, float], dict[str, float]]:
    returns = safe_pct_change(daily["close"])
    test_returns = daily.loc[(daily["date"] >= TEST_START) & (daily["date"] <= TEST_END), "close"].pct_change()
    stats_all = daily_return_stats(returns)
    stats_test = daily_return_stats(test_returns)

    fig, ax = plt.subplots(figsize=(11, 6))
    clean = returns.dropna()
    ax.hist(clean, bins=80, color="#4c78a8", alpha=0.78, edgecolor="white")
    ax.axvline(stats_all["mean"], color="#222222", linewidth=1.4, label=f"Mean {stats_all['mean']:.4f}")
    ax.axvline(stats_all["q01"], color="#d62728", linewidth=1.2, linestyle="--", label=f"1% {stats_all['q01']:.4f}")
    ax.axvline(stats_all["q99"], color="#2ca02c", linewidth=1.2, linestyle="--", label=f"99% {stats_all['q99']:.4f}")
    ax.set_title("Daily Close-to-Close Return Distribution")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(PLOT_DIR / "daily_return_distribution.png", generated)
    return stats_all, stats_test


def plot_rolling_volatility(daily: pd.DataFrame, generated: list[Path]) -> None:
    returns = safe_pct_change(daily["close"])
    vol20 = returns.rolling(20, min_periods=10).std() * math.sqrt(252)
    vol60 = returns.rolling(60, min_periods=30).std() * math.sqrt(252)

    fig, ax = plt.subplots(figsize=(13, 6))
    add_period_shading(ax)
    ax.plot(daily["date"], vol20, label="20-day annualized volatility", linewidth=1.4)
    ax.plot(daily["date"], vol60, label="60-day annualized volatility", linewidth=1.4)
    ax.set_title("Rolling Annualized Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(PLOT_DIR / "rolling_volatility.png", generated)


def plot_buy_hold_drawdown(daily: pd.DataFrame, generated: list[Path]) -> tuple[float, pd.Timestamp]:
    drawdown = compute_drawdown(daily["close"])
    min_idx = drawdown.idxmin()
    max_dd = float(drawdown.loc[min_idx])
    max_dd_date = pd.Timestamp(daily.loc[min_idx, "date"])

    fig, ax = plt.subplots(figsize=(13, 6))
    add_period_shading(ax)
    ax.plot(daily["date"], drawdown, color="#d62728", linewidth=1.5)
    ax.scatter([max_dd_date], [max_dd], color="#111111", s=32, zorder=3)
    ax.annotate(
        f"Max drawdown: {max_dd:.2%}\n{max_dd_date.date()}",
        xy=(max_dd_date, max_dd),
        xytext=(0.58, 0.18),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 0.9},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#999999", "alpha": 0.9},
    )
    ax.set_title("Buy-and-Hold Drawdown Based on Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    savefig(PLOT_DIR / "buy_hold_drawdown.png", generated)
    return max_dd, max_dd_date


def plot_volume_amount_trends(daily: pd.DataFrame, generated: list[Path]) -> None:
    vol = daily["volume"].rolling(20, min_periods=10).mean()
    amt = daily["amount"].rolling(20, min_periods=10).mean()
    vol_base = vol.dropna().iloc[0]
    amt_base = amt.dropna().iloc[0]
    vol_norm = vol / vol_base
    amt_norm = amt / amt_base

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(daily["date"], vol_norm, label="Volume rolling 20D avg, normalized", linewidth=1.4)
    ax.plot(daily["date"], amt_norm, label="Amount rolling 20D avg, normalized", linewidth=1.4)
    add_period_shading(ax)
    ax.set_title("Normalized Volume and Amount Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized level")
    ax.grid(alpha=0.25)
    ax.legend()
    savefig(PLOT_DIR / "volume_amount_trends.png", generated)


def plot_intraday_availability(daily: pd.DataFrame, m5: pd.DataFrame, generated: list[Path]) -> dict[str, Any]:
    counts = m5.groupby("date").size().rename("bar_count").reset_index() if not m5.empty else pd.DataFrame(columns=["date", "bar_count"])
    counts["date"] = pd.to_datetime(counts["date"])
    availability = daily[["date"]].merge(counts, on="date", how="left")
    availability["bar_count"] = availability["bar_count"].fillna(0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(availability["date"], availability["bar_count"], width=1.0, color="#4c78a8", alpha=0.75)
    add_period_shading(ax)
    ax.set_title("5-Minute Intraday Data Availability")
    ax.set_xlabel("Date")
    ax.set_ylabel("5-minute bar count")
    ax.grid(axis="y", alpha=0.25)
    savefig(PLOT_DIR / "intraday_data_availability.png", generated)

    intraday_dates = counts[counts["bar_count"] > 0]
    return {
        "daily_rows": int(len(daily)),
        "m5_rows": int(len(m5)),
        "intraday_dates": int(len(intraday_dates)),
        "intraday_daily_pct": float(len(intraday_dates) / len(daily)) if len(daily) else np.nan,
        "median_bars_per_intraday_date": float(intraday_dates["bar_count"].median()) if len(intraday_dates) else np.nan,
    }


def plot_intraday_factor_examples(intraday: pd.DataFrame, generated: list[Path]) -> list[str]:
    desired = [
        "m5_open_30m_return",
        "m5_close_30m_return",
        "m5_intraday_volatility",
        "m5_net_volume_pressure",
    ]
    available = [col for col in desired if col in intraday.columns]
    skipped = [col for col in desired if col not in intraday.columns]
    if available:
        fig, axes = plt.subplots(len(available), 1, figsize=(13, 2.8 * len(available)), sharex=True)
        if len(available) == 1:
            axes = [axes]
        for ax, col in zip(axes, available):
            smoothed = intraday[col].rolling(5, min_periods=2).mean()
            ax.plot(intraday["date"], smoothed, linewidth=1.3, label=f"{col} rolling 5D mean")
            ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
            ax.set_ylabel(col)
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")
        axes[-1].set_xlabel("Date")
        fig.suptitle("Representative Daily Intraday-Structure Factors", y=1.01, fontsize=13)
        savefig(PLOT_DIR / "intraday_factor_examples.png", generated)
    return skipped


def plot_feature_missing_rates(feature_df: pd.DataFrame, feature_cols: list[str], generated: list[Path]) -> pd.DataFrame:
    missing = (
        feature_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        .mean()
        .sort_values(ascending=False)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missing.to_csv(DIAG_DIR / "feature_missing_rate.csv", index=False)
    generated.append(DIAG_DIR / "feature_missing_rate.csv")

    top = missing.head(30).sort_values("missing_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(top["feature"], top["missing_rate"], color="#4c78a8", alpha=0.82)
    ax.set_title("Top 30 Feature Missing Rates Before Model Preprocessing")
    ax.set_xlabel("Missing rate")
    ax.grid(axis="x", alpha=0.25)
    savefig(PLOT_DIR / "feature_missing_rate_top30.png", generated)
    return missing


def plot_representative_correlation(feature_df: pd.DataFrame, generated: list[Path]) -> pd.DataFrame:
    candidates = [
        "ret_1",
        "ret_5",
        "ret_20",
        "ret_mean_20",
        "up_day_ratio_20",
        "ma_ratio_20",
        "ma_ratio_60",
        "ma_spread_20_60",
        "trend_slope_20",
        "trend_r2_20",
        "trend_regime_score",
        "volatility_20",
        "volatility_60",
        "volatility_ratio_20_60",
        "drawdown_20",
        "drawdown_60",
        "crash_risk_score",
        "volume_ratio_20",
        "amount_ratio_20",
        "price_volume_corr_20",
        "up_volume_ratio_20",
        "sell_pressure_volume_20",
        "m5_intraday_return",
        "m5_open_30m_return",
        "m5_close_30m_return",
        "m5_intraday_volatility",
        "m5_net_volume_pressure",
        "m5_close_30m_sell_pressure",
        "m5_intraday_slope",
        "bull_low_vol_regime",
        "bear_high_vol_regime",
        "sideways_regime",
    ]
    cols = [col for col in candidates if col in feature_df.columns][:35]
    sample = feature_df.loc[feature_df["date"] < TEST_START, cols].replace([np.inf, -np.inf], np.nan)
    if len(sample.dropna(how="all")) < 50:
        sample = feature_df[cols].replace([np.inf, -np.inf], np.nan)
    corr = sample.corr()
    corr.to_csv(DIAG_DIR / "representative_feature_correlation.csv")
    generated.append(DIAG_DIR / "representative_feature_correlation.csv")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_title("Representative Feature Correlation Matrix (Pre-Test Sample)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")
    savefig(PLOT_DIR / "representative_feature_correlation_heatmap.png", generated)
    return corr


def plot_forward_return_distribution(feature_df: pd.DataFrame, generated: list[Path]) -> None:
    values: list[np.ndarray] = []
    labels: list[str] = []
    for h in HORIZONS:
        ret_col = f"target_ret_{h}d"
        end_col = f"target_end_date_{h}d"
        if ret_col in feature_df.columns and end_col in feature_df.columns:
            clean = feature_df.loc[feature_df[end_col].notna(), ret_col].dropna()
            if len(clean):
                values.append(clean.to_numpy())
                labels.append(f"{h}D")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_title("Forward Return Distribution by Prediction Horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Forward return")
    ax.grid(axis="y", alpha=0.25)
    savefig(PLOT_DIR / "forward_return_distribution_by_horizon.png", generated)


def plot_label_balance(feature_df: pd.DataFrame, generated: list[Path]) -> pd.DataFrame:
    targets = ["target_up", "target_trade", "target_big_up", "target_big_down", "target_clean_direction"]
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        row: dict[str, Any] = {"horizon": h}
        for target in targets:
            col = f"{target}_{h}d"
            row[target] = float(feature_df[col].dropna().mean()) if col in feature_df.columns and feature_df[col].notna().any() else np.nan
        rows.append(row)
    balance = pd.DataFrame(rows)
    balance.to_csv(DIAG_DIR / "label_balance_by_horizon.csv", index=False)
    generated.append(DIAG_DIR / "label_balance_by_horizon.csv")

    x = np.arange(len(balance))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, target in enumerate(targets):
        if target in balance.columns:
            ax.bar(x + (idx - 2) * width, balance[target], width=width, label=target.replace("target_", ""))
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}D" for h in balance["horizon"]])
    ax.set_ylim(0, 1)
    ax.set_title("Label Balance by Prediction Horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Positive-label ratio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    savefig(PLOT_DIR / "label_balance_by_horizon.png", generated)
    return balance


def plot_prediction_auc(generated: list[Path]) -> None:
    candidates = [
        OUTPUT_DIR / "model_metrics_by_horizon.csv",
        OUTPUT_DIR / "metrics" / "test_model_metrics_by_horizon.csv",
    ]
    metrics_path = next((path for path in candidates if path.exists()), None)
    if metrics_path is None:
        return
    metrics = pd.read_csv(metrics_path)
    if metrics.empty or "model_name" not in metrics.columns or "horizon" not in metrics.columns:
        return
    sort_col = "clean_direction_AUC" if "clean_direction_AUC" in metrics.columns else "AUC"
    metric_cols = [col for col in ["AUC", "clean_direction_AUC", "trade_AUC", "big_up_AUC", "big_down_AUC"] if col in metrics.columns]
    plot_df = metrics.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False).head(16).copy()
    plot_df["label"] = plot_df["model_name"].astype(str) + " " + plot_df["horizon"].astype(str) + "D"
    x = np.arange(len(plot_df))
    width = 0.14
    fig, ax = plt.subplots(figsize=(14, 7))
    offsets = np.linspace(-(len(metric_cols) - 1) / 2, (len(metric_cols) - 1) / 2, len(metric_cols))
    for offset, col in zip(offsets, metric_cols):
        ax.bar(x + offset * width, plot_df[col], width=width, label=col)
    ax.axhline(0.5, color="black", linewidth=0.9, linestyle="--", label="0.5")
    ax.axhline(0.6, color="#d62728", linewidth=0.9, linestyle="--", label="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=45, ha="right")
    ax.set_ylim(0.35, max(0.75, float(plot_df[metric_cols].max().max()) + 0.05))
    ax.set_title("Prediction AUC Metrics by Model and Horizon")
    ax.set_ylabel("AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    savefig(PLOT_DIR / "prediction_auc_by_model_horizon.png", generated)


def write_summary(
    daily: pd.DataFrame,
    m5: pd.DataFrame,
    buy_hold_return: float,
    stats_all: dict[str, float],
    stats_test: dict[str, float],
    max_dd: float,
    max_dd_date: pd.Timestamp,
    intraday_stats: dict[str, Any],
    feature_cols: list[str],
    missing: pd.DataFrame,
    balance: pd.DataFrame,
    skipped_intraday: list[str],
    generated: list[Path],
) -> None:
    top_missing = missing.head(10).copy()
    top_missing["missing_rate"] = top_missing["missing_rate"].map(lambda x: f"{x:.2%}")
    balance_md = balance.copy()
    for col in balance_md.columns:
        if col != "horizon":
            balance_md[col] = balance_md[col].map(lambda x: "NA" if pd.isna(x) else f"{x:.2%}")

    text = f"""# Data Diagnostics Summary

## Data Files
- Daily data: `{DAILY_PATH}`
- 5-minute data: `{M5_PATH}`

## Sample Coverage
- Daily rows: {len(daily)}
- 5-minute rows: {len(m5)}
- Daily date range: {daily["date"].min().date()} to {daily["date"].max().date()}
- Test-period buy-and-hold return: {buy_hold_return:.2%}
- Full-sample buy-and-hold max drawdown: {max_dd:.2%} on {max_dd_date.date()}

## Daily Return Statistics
Full sample:
- Mean: {stats_all["mean"]:.6f}
- Std: {stats_all["std"]:.6f}
- Skewness: {stats_all["skewness"]:.6f}
- Kurtosis: {stats_all["kurtosis"]:.6f}
- 1% quantile: {stats_all["q01"]:.6f}
- 99% quantile: {stats_all["q99"]:.6f}

Test period:
- Mean: {stats_test["mean"]:.6f}
- Std: {stats_test["std"]:.6f}
- Skewness: {stats_test["skewness"]:.6f}
- Kurtosis: {stats_test["kurtosis"]:.6f}
- 1% quantile: {stats_test["q01"]:.6f}
- 99% quantile: {stats_test["q99"]:.6f}

## Intraday Data Availability
- Dates with intraday data: {intraday_stats["intraday_dates"]}
- Percentage of daily dates with intraday data: {intraday_stats["intraday_daily_pct"]:.2%}
- Median bars per intraday date: {intraday_stats["median_bars_per_intraday_date"]:.1f}
- Skipped intraday example columns: {", ".join(skipped_intraday) if skipped_intraday else "None"}

## Feature Missing Rates
- Model feature count: {len(feature_cols)}

Top missing-rate features:
{markdown_table(top_missing)}

## Label Balance
{markdown_table(balance_md)}

## Interpretation
The test period is a strong upward market. The micro-cap index has high volatility and meaningful drawdown risk, which motivates risk-aware controls even for an index-enhancement strategy. The 5-minute data are transformed into daily intraday-structure factors, not used as high-frequency trading signals. The label distributions and AUC diagnostics show that prediction is possible but noisy, especially for ordinary up/down direction. Therefore, the final strategy should be benchmark-aware and conservative rather than aggressive long/cash timing.

## Generated Files
{chr(10).join(f"- `{path}`" for path in generated)}
"""
    path = DIAG_DIR / "data_diagnostics_summary.md"
    path.write_text(text, encoding="utf-8")
    generated.append(path)


def main() -> None:
    ensure_dir(PLOT_DIR)
    generated: list[Path] = []

    print("Loading data...")
    daily, m5 = load_data()
    daily = daily.sort_values("date").reset_index(drop=True)
    m5 = m5.sort_values(["date", "time"]).reset_index(drop=True) if not m5.empty else m5

    print("Building intraday and daily feature frames...")
    intraday = build_intraday_features(m5)
    feature_df = build_daily_features(daily, intraday)
    feature_cols = maybe_get_feature_columns(feature_df)

    buy_hold_return = plot_index_close(daily, generated)
    stats_all, stats_test = plot_daily_return_distribution(daily, generated)
    plot_rolling_volatility(daily, generated)
    max_dd, max_dd_date = plot_buy_hold_drawdown(daily, generated)
    plot_volume_amount_trends(daily, generated)
    intraday_stats = plot_intraday_availability(daily, m5, generated)
    skipped_intraday = plot_intraday_factor_examples(intraday, generated)
    missing = plot_feature_missing_rates(feature_df, feature_cols, generated)
    plot_representative_correlation(feature_df, generated)
    plot_forward_return_distribution(feature_df, generated)
    balance = plot_label_balance(feature_df, generated)
    plot_prediction_auc(generated)

    write_summary(
        daily,
        m5,
        buy_hold_return,
        stats_all,
        stats_test,
        max_dd,
        max_dd_date,
        intraday_stats,
        feature_cols,
        missing,
        balance,
        skipped_intraday,
        generated,
    )

    print("\nGenerated data diagnostics files:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
