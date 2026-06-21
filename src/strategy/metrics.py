from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import INITIAL_CAPITAL


def annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / 252
    if years <= 0 or total <= 0:
        return np.nan
    return float(total ** (1 / years) - 1)


def detect_benchmark_clone(df: pd.DataFrame, excess_return: float, tracking_error: float) -> bool:
    position_gap = (df["final_position"] - 1.0).abs()
    avg_gap = float(position_gap.mean())
    reduced_days = int((df["final_position"] < 0.995).sum())
    reduced_ratio = reduced_days / max(len(df), 1)
    return (
        avg_gap < 0.006
        and abs(excess_return) < 0.003
        and (not math.isfinite(tracking_error) or tracking_error < 0.006)
        and reduced_ratio < 0.03
    )


def compute_strategy_metrics(strategy_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    benchmark = strategy_daily.loc[strategy_daily["strategy"] == "buy_hold"].copy()
    benchmark_total = (
        benchmark["buy_hold_equity"].iloc[-1] / INITIAL_CAPITAL - 1 if not benchmark.empty else np.nan
    )
    benchmark_mdd = benchmark["buy_hold_drawdown"].min() if not benchmark.empty else np.nan

    for strategy, grp in strategy_daily.groupby("strategy", sort=True):
        df = grp.sort_values("date").copy()
        ret = df["strategy_return"]
        equity = df["equity"]
        total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1) if len(equity) else np.nan
        ann_return = annualized_return(equity)
        ann_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else np.nan
        sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else np.nan
        mdd = float(df["drawdown"].min()) if len(df) else np.nan
        active = df["active_return"]
        buy_signal_count = int((df["signal"] == "BUY").sum()) if "signal" in df.columns else 0
        sell_signal_count = int((df["signal"] == "SELL").sum()) if "signal" in df.columns else 0
        hold_signal_count = int((df["signal"] == "HOLD").sum()) if "signal" in df.columns else 0
        buy_rows = df.loc[df["signal"] == "BUY"] if "signal" in df.columns else pd.DataFrame()
        sell_rows = df.loc[df["signal"] == "SELL"] if "signal" in df.columns else pd.DataFrame()
        position_gap = (1.0 - df["final_position"]).clip(lower=0.0)
        missed_upside = float((position_gap * df["buy_hold_return"].clip(lower=0.0)).sum())
        avoided_downside = float((position_gap * (-df["buy_hold_return"].clip(upper=0.0))).sum())
        tracking_error = float(active.std() * np.sqrt(252)) if len(active) > 1 else np.nan
        annualized_excess = float(active.mean() * 252) if len(active) else np.nan
        excess_return = float(total_return - benchmark_total) if math.isfinite(total_return) and math.isfinite(benchmark_total) else np.nan
        info_ratio = annualized_excess / tracking_error if tracking_error and tracking_error > 0 else np.nan
        benchmark_clone = True if strategy == "buy_hold" else detect_benchmark_clone(df, excess_return, tracking_error)

        rows.append(
            {
                "strategy": strategy,
                "total_return": total_return,
                "annualized_return": ann_return,
                "annualized_volatility": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "calmar": ann_return / abs(mdd) if mdd < 0 and math.isfinite(ann_return) else np.nan,
                "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
                "average_position": float(df["final_position"].mean()),
                "minimum_position": float(df["final_position"].min()),
                "maximum_position": float(df["final_position"].max()),
                "days_below_full_exposure": int((df["final_position"] < 0.995).sum()),
                "total_turnover": float(df["turnover"].sum()),
                "total_transaction_cost": float(df["transaction_cost"].sum()),
                "buy_signal_count": buy_signal_count,
                "sell_signal_count": sell_signal_count,
                "hold_signal_count": hold_signal_count,
                "win_rate_after_buy": float((buy_rows["buy_hold_return"] > 0).mean()) if not buy_rows.empty else np.nan,
                "successful_sell_count": int((sell_rows["buy_hold_return"] <= 0).sum()) if not sell_rows.empty else 0,
                "failed_sell_count": int((sell_rows["buy_hold_return"] > 0).sum()) if not sell_rows.empty else 0,
                "missed_upside": missed_upside,
                "avoided_downside": avoided_downside,
                "net_timing_contribution": avoided_downside - missed_upside - float(df["transaction_cost"].sum()),
                "excess_return_vs_buy_hold": excess_return,
                "annualized_excess_return": annualized_excess,
                "tracking_error": tracking_error,
                "information_ratio": info_ratio,
                "benchmark_total_return": benchmark_total,
                "benchmark_max_drawdown": benchmark_mdd,
                "benchmark_clone": benchmark_clone,
                "avg_abs_position_gap_from_1": float((df["final_position"] - 1.0).abs().mean()),
                "reduced_exposure_day_ratio": float((df["final_position"] < 0.995).mean()),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["strategy"]).reset_index(drop=True)


def validation_strategy_score(row: pd.Series) -> float:
    if bool(row.get("benchmark_clone", False)):
        return -1e6
    excess = float(row.get("excess_return_vs_buy_hold", 0.0) or 0.0)
    drawdown_improvement = float(row.get("benchmark_max_drawdown", 0.0) or 0.0) - float(row.get("max_drawdown", 0.0) or 0.0)
    sharpe = float(row.get("sharpe", 0.0) or 0.0)
    info_ratio = float(row.get("information_ratio", 0.0) or 0.0)
    turnover = float(row.get("total_turnover", 0.0) or 0.0)
    gap = float(row.get("avg_abs_position_gap_from_1", 0.0) or 0.0)
    clone_like_penalty = 0.10 if gap < 0.01 else 0.0
    return (
        2.0 * excess
        + 1.2 * drawdown_improvement
        + 0.08 * sharpe
        + 0.04 * info_ratio
        - 0.015 * turnover
        - clone_like_penalty
    )


def conservative_validation_strategy_score(row: pd.Series) -> float:
    if bool(row.get("benchmark_clone", False)):
        return -1e6
    excess = float(row.get("excess_return_vs_buy_hold", 0.0) or 0.0)
    drawdown_improvement = float(row.get("benchmark_max_drawdown", 0.0) or 0.0) - float(row.get("max_drawdown", 0.0) or 0.0)
    missed = float(row.get("missed_upside", 0.0) or 0.0)
    avoided = float(row.get("avoided_downside", 0.0) or 0.0)
    turnover = float(row.get("total_turnover", 0.0) or 0.0)
    avg_position = float(row.get("average_position", 1.0) or 1.0)
    reduced_ratio = float(row.get("reduced_exposure_day_ratio", 0.0) or 0.0)

    penalty = 0.0
    if avg_position < 0.975:
        penalty += 0.60 * (0.975 - avg_position)
    if reduced_ratio > 0.35:
        penalty += 0.25 * (reduced_ratio - 0.35)
    if missed > avoided:
        penalty += 1.25 * (missed - avoided)

    return (
        3.0 * excess
        + 1.4 * drawdown_improvement
        - 1.0 * missed
        + 0.7 * avoided
        - 0.02 * turnover
        + 0.20 * avg_position
        - penalty
    )


def ultra_conservative_validation_strategy_score(row: pd.Series) -> float:
    if bool(row.get("benchmark_clone", False)):
        return -1e6
    excess = float(row.get("excess_return_vs_buy_hold", 0.0) or 0.0)
    drawdown_improvement = float(row.get("benchmark_max_drawdown", 0.0) or 0.0) - float(row.get("max_drawdown", 0.0) or 0.0)
    missed = float(row.get("missed_upside", 0.0) or 0.0)
    avoided = float(row.get("avoided_downside", 0.0) or 0.0)
    turnover = float(row.get("total_turnover", 0.0) or 0.0)
    avg_position = float(row.get("average_position", 1.0) or 1.0)
    reduced_ratio = float(row.get("reduced_exposure_day_ratio", 0.0) or 0.0)

    penalty = 0.0
    if avg_position < 0.99:
        penalty += 1.2 * (0.99 - avg_position)
    if reduced_ratio > 0.25:
        penalty += 0.6 * (reduced_ratio - 0.25)
    if missed > avoided:
        penalty += 1.6 * (missed - avoided)
    if missed > 0.03:
        penalty += 2.0 * (missed - 0.03)

    return (
        4.0 * excess
        + 1.2 * drawdown_improvement
        - 1.8 * missed
        + 0.9 * avoided
        - 0.018 * turnover
        + 0.35 * avg_position
        - penalty
    )


def select_strategy_rows(metrics: pd.DataFrame) -> dict[str, pd.Series | None]:
    buy_hold = metrics.loc[metrics["strategy"] == "buy_hold"]
    no_leverage = metrics.loc[
        metrics["strategy"].isin(
            [
                "ml_big_up_index_enhancement",
                "ml_conservative_full_participation_enhancement",
                "ml_ultra_conservative_full_participation_enhancement",
                "ml_direct_signal_timing",
                "ml_upside_participation_enhancement",
                "ml_stability_aware_high_participation",
                "ml_risk_budget_enhancement",
                "ml_early_stress_risk_budget",
                "ml_external_risk_budget_enhancement_index_only",
                "ml_external_risk_budget_enhancement_index_margin",
                "ml_external_risk_budget_enhancement_index_shibor",
                "ml_external_risk_budget_enhancement_all",
                "ml_external_early_stress_risk_budget_index_only",
                "ml_external_early_stress_risk_budget_all",
                "ml_external_soft_confirm_risk_budget",
                "ml_external_soft_confirm_risk_budget_index_only",
                "ml_external_soft_confirm_early_stress",
                "ml_external_soft_confirm_early_stress_index_only",
                "ml_external_soft_lite_risk_budget",
                "ml_external_soft_lite_risk_budget_index_only",
                "ml_external_soft_lite_early_stress",
                "ml_external_soft_lite_early_stress_index_only",
            ]
        )
    ].copy()
    real_no_leverage = no_leverage.loc[~no_leverage["benchmark_clone"].astype(bool)].copy()
    plus = metrics.loc[metrics["strategy"].isin(["ml_big_up_plus_115", "ml_big_up_plus_120"])].copy()

    best_no_lev = (
        no_leverage.sort_values("total_return", ascending=False).iloc[0]
        if not no_leverage.empty
        else None
    )
    best_real = (
        real_no_leverage.sort_values("total_return", ascending=False).iloc[0]
        if not real_no_leverage.empty
        else best_no_lev
    )
    best_plus = plus.sort_values("total_return", ascending=False).iloc[0] if not plus.empty else None
    return {
        "buy_hold": buy_hold.iloc[0] if not buy_hold.empty else None,
        "best_no_leverage": best_no_lev,
        "best_real_no_leverage": best_real,
        "best_enhanced_exposure": best_plus,
    }
