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
    """计算策略与买入持有的收益、回撤、换手与归因指标。"""
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
    """旧版单验证集评分，保留给现有脚本兼容使用。"""
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
    """旧版保守策略单验证集评分，保留给现有脚本兼容使用。"""
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
    """旧版超保守策略单验证集评分，保留给现有脚本兼容使用。"""
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


def robust_score_from_fold_metrics(fold_metrics: pd.DataFrame) -> float:
    """基于多验证折综合评估收益、回撤、换手、机会成本与稳定性。"""
    return float(robust_score_details_from_fold_metrics(fold_metrics)["robust_score"])


def robust_score_details_from_fold_metrics(fold_metrics: pd.DataFrame) -> dict[str, float]:
    """返回稳健评分及其组成项，便于实验报告解释分数来源。"""
    if fold_metrics.empty:
        return {
            "robust_score": -1e9,
            "avg_excess_return": 0.0,
            "avg_sharpe": 0.0,
            "avg_calmar": 0.0,
            "avg_turnover": 0.0,
            "avg_max_drawdown": 0.0,
            "avg_position": 0.0,
            "avg_missed_upside": 0.0,
            "avg_avoided_downside": 0.0,
            "excess_stability": 0.0,
            "positive_fold_ratio": 0.0,
            "excess_std": 0.0,
        }
    work = fold_metrics.copy()
    if "strategy" in work.columns:
        work = work.loc[work["strategy"] != "buy_hold"].copy()
    if work.empty:
        return {
            "robust_score": -1e9,
            "avg_excess_return": 0.0,
            "avg_sharpe": 0.0,
            "avg_calmar": 0.0,
            "avg_turnover": 0.0,
            "avg_max_drawdown": 0.0,
            "avg_position": 0.0,
            "avg_missed_upside": 0.0,
            "avg_avoided_downside": 0.0,
            "excess_stability": 0.0,
            "positive_fold_ratio": 0.0,
            "excess_std": 0.0,
        }

    excess = work["excess_return_vs_buy_hold"].fillna(0.0)
    sharpe = work["sharpe"].fillna(0.0)
    calmar = work["calmar"].fillna(0.0)
    turnover = work["total_turnover"].fillna(0.0)
    max_drawdown = work["max_drawdown"].fillna(0.0)
    avg_position = work["average_position"].fillna(0.0) if "average_position" in work.columns else pd.Series(0.0, index=work.index)
    missed_upside = work["missed_upside"].fillna(0.0) if "missed_upside" in work.columns else pd.Series(0.0, index=work.index)
    avoided_downside = work["avoided_downside"].fillna(0.0) if "avoided_downside" in work.columns else pd.Series(0.0, index=work.index)
    reduced_ratio = (
        work["reduced_exposure_day_ratio"].fillna(0.0)
        if "reduced_exposure_day_ratio" in work.columns
        else pd.Series(0.0, index=work.index)
    )
    excess_std = float(excess.std(ddof=0))
    stability = 1.0 / (1.0 + excess_std)
    positive_fold_ratio = float((excess > 0).mean()) if len(excess) else 0.0
    downside_penalty = float((excess.clip(upper=0.0).abs()).mean()) if len(excess) else 0.0
    upside_opportunity_cost = float(missed_upside.mean())
    downside_capture = float(avoided_downside.mean())
    near_full_reward = float(np.clip(avg_position.mean() - 0.94, 0.0, 0.08))
    effective_defense_reward = float(np.clip(downside_capture - 0.5 * upside_opportunity_cost, -0.2, 0.2))
    over_defensive_penalty = float(np.clip(reduced_ratio.mean() - 0.30, 0.0, 1.0))
    negative_excess_penalty = float(np.clip(-excess.mean(), 0.0, 1.0))

    robust_score = float(
        5.5 * excess.mean()
        + 0.20 * sharpe.mean()
        + 0.12 * calmar.mean()
        - 1.3 * abs(max_drawdown.mean())
        - 0.02 * turnover.mean()
        + 0.9 * stability
        + 0.6 * positive_fold_ratio
        - 1.5 * downside_penalty
        - 3.4 * upside_opportunity_cost
        + 1.2 * downside_capture
        + 2.6 * near_full_reward
        + 0.8 * effective_defense_reward
        - 0.8 * over_defensive_penalty
        - 1.5 * excess_std
        - 6.0 * negative_excess_penalty
    )
    return {
        "robust_score": robust_score,
        "avg_excess_return": float(excess.mean()),
        "avg_sharpe": float(sharpe.mean()),
        "avg_calmar": float(calmar.mean()),
        "avg_turnover": float(turnover.mean()),
        "avg_max_drawdown": float(max_drawdown.mean()),
        "avg_position": float(avg_position.mean()),
        "avg_missed_upside": float(missed_upside.mean()),
        "avg_avoided_downside": float(avoided_downside.mean()),
        "excess_stability": float(stability),
        "positive_fold_ratio": positive_fold_ratio,
        "excess_std": excess_std,
    }
