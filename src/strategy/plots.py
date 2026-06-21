from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PLOT_DIR
from src.strategy.upside_participation import (
    EARLY_STRESS_STRATEGY_NAME,
    RISK_BUDGET_STRATEGY_NAME,
    STABILITY_AWARE_STRATEGY_NAME,
    UPSIDE_STRATEGY_NAME,
)


def _strategy_frame(strategy_daily: pd.DataFrame, strategy: str) -> pd.DataFrame:
    return strategy_daily.loc[strategy_daily["strategy"] == strategy].sort_values("date").copy()


def plot_strategy_outputs(strategy_daily: pd.DataFrame, metrics: pd.DataFrame, main_strategy: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    buy_hold = _strategy_frame(strategy_daily, "buy_hold")
    main = _strategy_frame(strategy_daily, main_strategy)
    if buy_hold.empty or main.empty:
        return

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_equity"], label="Buy-and-hold", linewidth=1.8)
    plt.plot(main["trade_date"], main["equity"], label=main_strategy, linewidth=1.8)
    plt.title("Equity Curve: Buy-and-Hold vs Main No-Leverage ML Strategy")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_buyhold_vs_main_ml_strategy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_drawdown"], label="Buy-and-hold", linewidth=1.8)
    plt.plot(main["trade_date"], main["drawdown"], label=main_strategy, linewidth=1.8)
    plt.title("Drawdown Curve: Buy-and-Hold vs Main No-Leverage ML Strategy")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_buyhold_vs_main_ml_strategy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(main["trade_date"], main["final_position"], label="Final position", linewidth=1.6)
    plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Main Strategy Position Exposure Over Time")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.ylim(0, max(1.05, main["final_position"].max() + 0.05))
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "position_exposure_main_ml_strategy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(main["trade_date"], main["pred_big_up_prob"], label="Predicted big-up probability", linewidth=1.5)
    plt.plot(main["trade_date"], main["final_position"], label="Final position", linewidth=1.5)
    plt.title("Predicted Big-Up Probability and Main Strategy Position")
    plt.xlabel("Date")
    plt.ylabel("Probability / Exposure")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "predicted_big_up_probability_and_position.png", dpi=160)
    plt.close()

    plus_frames = [
        _strategy_frame(strategy_daily, "ml_big_up_plus_115"),
        _strategy_frame(strategy_daily, "ml_big_up_plus_120"),
    ]
    if any(not frame.empty for frame in plus_frames):
        plt.figure(figsize=(13, 6))
        plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_equity"], label="Buy-and-hold", linewidth=1.8)
        for frame in plus_frames:
            if not frame.empty:
                plt.plot(frame["trade_date"], frame["equity"], label=frame["strategy"].iloc[0], linewidth=1.5)
        plt.title("Enhanced-Exposure Strategies vs Buy-and-Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity (RMB)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "equity_curve_enhanced_exposure_vs_buyhold.png", dpi=160)
        plt.close()

    plot_metrics = metrics.sort_values("total_return", ascending=False)
    plt.figure(figsize=(11, 6))
    plt.bar(plot_metrics["strategy"], plot_metrics["total_return"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Total Return")
    plt.title("Strategy Return Comparison")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "strategy_return_comparison.png", dpi=160)
    plt.close()


def plot_signal_quantile_stability(
    bucket_df: pd.DataFrame,
    selection: dict,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    selected = selection["selected"]
    sample = bucket_df.loc[
        (bucket_df["model_name"] == selected["model_name"])
        & (bucket_df["horizon"] == int(selected["horizon"]))
        & (bucket_df["signal_name"] == selected["signal_name"])
    ].copy()
    if sample.empty:
        return
    for period, filename in [
        ("validation", "signal_quantile_stability_validation.png"),
        ("test", "signal_quantile_stability_test.png"),
    ]:
        period_df = sample.loc[sample["period"] == period].sort_values("quantile_index")
        if period_df.empty:
            continue
        plt.figure(figsize=(9, 5))
        colors = ["#6b7280"] * len(period_df)
        if len(colors):
            colors[0] = "#9ca3af"
            colors[-1] = "#2563eb"
        plt.bar(period_df["quantile_group"], period_df["mean_forward_return"], color=colors)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(
            f"{period.title()} Signal Quantile Stability: "
            f"{selected['signal_name']} / {selected['model_name']} / {selected['horizon']}D"
        )
        plt.xlabel("Signal quantile")
        plt.ylabel("Mean next-day return")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / filename, dpi=160)
        plt.close()


def _previous_best_real_no_leverage(metrics: pd.DataFrame) -> str | None:
    candidates = metrics.loc[
        metrics["strategy"].isin(
            [
                "ml_big_up_index_enhancement",
                "ml_conservative_full_participation_enhancement",
                "ml_ultra_conservative_full_participation_enhancement",
                "ml_direct_signal_timing",
            ]
        )
    ].copy()
    if "benchmark_clone" in candidates.columns:
        candidates = candidates.loc[~candidates["benchmark_clone"].astype(bool)]
    if candidates.empty:
        return None
    return str(candidates.sort_values("total_return", ascending=False).iloc[0]["strategy"])


def _current_best_real_no_leverage(metrics: pd.DataFrame) -> str | None:
    candidates = metrics.loc[
        metrics["strategy"].isin(
            [
                "ml_big_up_index_enhancement",
                "ml_conservative_full_participation_enhancement",
                "ml_ultra_conservative_full_participation_enhancement",
                "ml_direct_signal_timing",
                UPSIDE_STRATEGY_NAME,
                STABILITY_AWARE_STRATEGY_NAME,
                RISK_BUDGET_STRATEGY_NAME,
                EARLY_STRESS_STRATEGY_NAME,
            ]
        )
    ].copy()
    if "benchmark_clone" in candidates.columns:
        candidates = candidates.loc[~candidates["benchmark_clone"].astype(bool)]
    if candidates.empty:
        return None
    return str(candidates.sort_values("total_return", ascending=False).iloc[0]["strategy"])


def plot_upside_participation_outputs(
    strategy_daily: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_signal_name: str,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    buy_hold = _strategy_frame(strategy_daily, "buy_hold")
    upside = _strategy_frame(strategy_daily, UPSIDE_STRATEGY_NAME)
    if buy_hold.empty or upside.empty:
        return
    previous_best_name = _previous_best_real_no_leverage(metrics)
    previous = _strategy_frame(strategy_daily, previous_best_name) if previous_best_name else pd.DataFrame()

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_equity"], label="Buy-and-hold", linewidth=1.8)
    if not previous.empty:
        plt.plot(previous["trade_date"], previous["equity"], label=f"Previous best: {previous_best_name}", linewidth=1.5)
    plt.plot(upside["trade_date"], upside["equity"], label=UPSIDE_STRATEGY_NAME, linewidth=1.8)
    plt.title("Equity Curve: Upside Participation vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_upside_participation_vs_buyhold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_drawdown"], label="Buy-and-hold", linewidth=1.8)
    if not previous.empty:
        plt.plot(previous["trade_date"], previous["drawdown"], label=f"Previous best: {previous_best_name}", linewidth=1.5)
    plt.plot(upside["trade_date"], upside["drawdown"], label=UPSIDE_STRATEGY_NAME, linewidth=1.8)
    plt.title("Drawdown Curve: Upside Participation vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_upside_participation_vs_buyhold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(upside["trade_date"], upside["final_position"], label="Final position", linewidth=1.6)
    plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Upside Participation Position Exposure")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.ylim(0.84, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "position_exposure_upside_participation.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    if selected_signal_name in upside.columns:
        axes[0].plot(upside["trade_date"], upside[selected_signal_name], label=selected_signal_name, linewidth=1.4)
    else:
        axes[0].plot(upside["trade_date"], np.nan, label=selected_signal_name, linewidth=1.4)
    axes[0].set_ylabel("Signal")
    axes[0].legend(loc="upper left")
    axes[1].plot(upside["trade_date"], upside["pred_ret"], label="pred_ret", color="#2563eb", linewidth=1.3)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Pred ret")
    axes[1].legend(loc="upper left")
    axes[2].plot(upside["trade_date"], upside["final_position"], label="final_position", color="#059669", linewidth=1.4)
    axes[2].set_ylabel("Position")
    axes[2].set_ylim(0.84, 1.02)
    axes[2].legend(loc="upper left")
    axes[3].bar(upside["trade_date"], upside["fwd_ret_1d"], label="fwd_ret_1d", color="#6b7280", width=1.0)
    axes[3].axhline(0, color="black", linewidth=0.7)
    axes[3].set_ylabel("Next-day ret")
    axes[3].legend(loc="upper left")
    axes[3].set_xlabel("Date")
    fig.suptitle("Upside Signal and Position")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "upside_signal_and_position.png", dpi=160)
    plt.close(fig)


def plot_stability_aware_high_participation_outputs(
    strategy_daily: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_components: list[dict],
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    buy_hold = _strategy_frame(strategy_daily, "buy_hold")
    stability = _strategy_frame(strategy_daily, STABILITY_AWARE_STRATEGY_NAME)
    if buy_hold.empty or stability.empty:
        return
    current_best_name = _current_best_real_no_leverage(metrics)
    previous = (
        _strategy_frame(strategy_daily, current_best_name)
        if current_best_name and current_best_name != STABILITY_AWARE_STRATEGY_NAME
        else pd.DataFrame()
    )

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_equity"], label="Buy-and-hold", linewidth=1.8)
    if not previous.empty:
        plt.plot(previous["trade_date"], previous["equity"], label=f"Current best: {current_best_name}", linewidth=1.5)
    plt.plot(stability["trade_date"], stability["equity"], label=STABILITY_AWARE_STRATEGY_NAME, linewidth=1.8)
    plt.title("Equity Curve: Stability-Aware High Participation vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_stability_aware_high_participation_vs_buyhold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_drawdown"], label="Buy-and-hold", linewidth=1.8)
    if not previous.empty:
        plt.plot(previous["trade_date"], previous["drawdown"], label=f"Current best: {current_best_name}", linewidth=1.5)
    plt.plot(stability["trade_date"], stability["drawdown"], label=STABILITY_AWARE_STRATEGY_NAME, linewidth=1.8)
    plt.title("Drawdown Curve: Stability-Aware High Participation vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_stability_aware_high_participation_vs_buyhold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(stability["trade_date"], stability["final_position"], label="Final position", linewidth=1.6)
    plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Stability-Aware High-Participation Exposure")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.ylim(0.90, 1.01)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "position_exposure_stability_aware_high_participation.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(stability["trade_date"], stability["ensemble_upside_score"], label="ensemble_upside_score", linewidth=1.4)
    axes[0].set_ylabel("Ensemble")
    axes[0].legend(loc="upper left")
    rank_cols = [col for col in stability.columns if col.startswith("component_") and col.endswith("_rank")]
    if rank_cols:
        axes[1].plot(stability["trade_date"], stability[rank_cols[0]], label=rank_cols[0], color="#2563eb", linewidth=1.2)
    axes[1].set_ylabel("Top rank")
    axes[1].legend(loc="upper left")
    axes[2].plot(stability["trade_date"], stability["final_position"], label="final_position", color="#059669", linewidth=1.4)
    axes[2].set_ylim(0.90, 1.01)
    axes[2].set_ylabel("Position")
    axes[2].legend(loc="upper left")
    axes[3].bar(stability["trade_date"], stability["fwd_ret_1d"], label="fwd_ret_1d", color="#6b7280", width=1.0)
    axes[3].axhline(0, color="black", linewidth=0.7)
    axes[3].set_ylabel("Next-day ret")
    axes[3].legend(loc="upper left")
    axes[3].set_xlabel("Date")
    fig.suptitle("Stability-Aware Signal Diagnostics")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "stability_aware_signal_diagnostics.png", dpi=160)
    plt.close(fig)

    if selected_components:
        labels = [f"{c['model_name']} {c['horizon']}D\n{c['signal_name']}" for c in selected_components]
        raw_scores = [float(c.get("raw_stable_score", 0.0) or 0.0) for c in selected_components]
        penalties = [float(c.get("complexity_penalty", 0.0) or 0.0) for c in selected_components]
        stable_scores = [float(c.get("stable_score", 0.0) or 0.0) for c in selected_components]
        x = np.arange(len(labels))
        width = 0.25
        plt.figure(figsize=(11, 5))
        plt.bar(x - width, raw_scores, width=width, label="raw score")
        plt.bar(x, penalties, width=width, label="penalty")
        plt.bar(x + width, stable_scores, width=width, label="stable score")
        plt.xticks(x, labels, rotation=15, ha="right")
        plt.ylabel("Score")
        plt.title("Stability-Aware Component Scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "stability_aware_component_scores.png", dpi=160)
        plt.close()


def plot_risk_budget_outputs(
    strategy_daily: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    buy_hold = _strategy_frame(strategy_daily, "buy_hold")
    stability = _strategy_frame(strategy_daily, STABILITY_AWARE_STRATEGY_NAME)
    risk_budget = _strategy_frame(strategy_daily, RISK_BUDGET_STRATEGY_NAME)
    if buy_hold.empty or risk_budget.empty:
        return

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_equity"], label="Buy-and-hold", linewidth=1.8)
    if not stability.empty:
        plt.plot(stability["trade_date"], stability["equity"], label=STABILITY_AWARE_STRATEGY_NAME, linewidth=1.5)
    plt.plot(risk_budget["trade_date"], risk_budget["equity"], label=RISK_BUDGET_STRATEGY_NAME, linewidth=1.8)
    plt.title("Equity Curve: Risk-Budget Enhancement vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_risk_budget_vs_buyhold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_drawdown"], label="Buy-and-hold", linewidth=1.8)
    if not stability.empty:
        plt.plot(stability["trade_date"], stability["drawdown"], label=STABILITY_AWARE_STRATEGY_NAME, linewidth=1.5)
    plt.plot(risk_budget["trade_date"], risk_budget["drawdown"], label=RISK_BUDGET_STRATEGY_NAME, linewidth=1.8)
    plt.title("Drawdown Curve: Risk-Budget Enhancement vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_risk_budget_vs_buyhold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(risk_budget["trade_date"], risk_budget["final_position"], label="Final position", linewidth=1.6)
    plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Risk-Budget Position Exposure")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.ylim(0.68, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "position_exposure_risk_budget.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    if "risk_score" in risk_budget.columns:
        axes[0].plot(risk_budget["trade_date"], risk_budget["risk_score"], label="risk_score", linewidth=1.4)
    axes[0].set_ylabel("Risk score")
    axes[0].legend(loc="upper left")
    axes[1].plot(risk_budget["trade_date"], risk_budget["final_position"], label="final_position", color="#059669", linewidth=1.4)
    axes[1].set_ylim(0.68, 1.02)
    axes[1].set_ylabel("Position")
    axes[1].legend(loc="upper left")
    if "pred_up_prob" in risk_budget.columns:
        axes[2].plot(risk_budget["trade_date"], risk_budget["pred_up_prob"], label="pred_up_prob", color="#2563eb", linewidth=1.2)
    axes[2].set_ylabel("Pred up")
    axes[2].legend(loc="upper left")
    axes[3].bar(risk_budget["trade_date"], risk_budget["fwd_ret_1d"], label="fwd_ret_1d", color="#6b7280", width=1.0)
    axes[3].axhline(0, color="black", linewidth=0.7)
    axes[3].set_ylabel("Next-day ret")
    axes[3].legend(loc="upper left")
    axes[3].set_xlabel("Date")
    fig.suptitle("Risk-Budget Signal Diagnostics")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "risk_budget_signal_diagnostics.png", dpi=160)
    plt.close(fig)


def plot_early_stress_risk_budget_outputs(
    strategy_daily: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    buy_hold = _strategy_frame(strategy_daily, "buy_hold")
    stability = _strategy_frame(strategy_daily, STABILITY_AWARE_STRATEGY_NAME)
    risk_budget = _strategy_frame(strategy_daily, RISK_BUDGET_STRATEGY_NAME)
    early = _strategy_frame(strategy_daily, EARLY_STRESS_STRATEGY_NAME)
    if buy_hold.empty or early.empty:
        return

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_equity"], label="Buy-and-hold", linewidth=1.8)
    if not stability.empty:
        plt.plot(stability["trade_date"], stability["equity"], label=STABILITY_AWARE_STRATEGY_NAME, linewidth=1.4)
    if not risk_budget.empty:
        plt.plot(risk_budget["trade_date"], risk_budget["equity"], label=RISK_BUDGET_STRATEGY_NAME, linewidth=1.4)
    plt.plot(early["trade_date"], early["equity"], label=EARLY_STRESS_STRATEGY_NAME, linewidth=1.8)
    plt.title("Equity Curve: Early-Stress Risk-Budget vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (RMB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "equity_curve_early_stress_risk_budget_vs_buyhold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.plot(buy_hold["trade_date"], buy_hold["buy_hold_drawdown"], label="Buy-and-hold", linewidth=1.8)
    if not stability.empty:
        plt.plot(stability["trade_date"], stability["drawdown"], label=STABILITY_AWARE_STRATEGY_NAME, linewidth=1.4)
    if not risk_budget.empty:
        plt.plot(risk_budget["trade_date"], risk_budget["drawdown"], label=RISK_BUDGET_STRATEGY_NAME, linewidth=1.4)
    plt.plot(early["trade_date"], early["drawdown"], label=EARLY_STRESS_STRATEGY_NAME, linewidth=1.8)
    plt.title("Drawdown Curve: Early-Stress Risk-Budget vs Buy-and-Hold")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "drawdown_curve_early_stress_risk_budget_vs_buyhold.png", dpi=160)
    plt.close()

    plt.figure(figsize=(13, 5))
    plt.plot(early["trade_date"], early["final_position"], label="Final position", linewidth=1.6)
    plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Early-Stress Risk-Budget Position Exposure")
    plt.xlabel("Date")
    plt.ylabel("Exposure")
    plt.ylim(0.78, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "position_exposure_early_stress_risk_budget.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    axes[0].plot(early["trade_date"], early["early_stress_score"], label="early_stress_score", linewidth=1.4)
    axes[0].set_ylabel("Stress")
    axes[0].legend(loc="upper left")
    axes[1].plot(early["trade_date"], early["pred_up_prob"], label="pred_up_prob", color="#2563eb", linewidth=1.2)
    axes[1].axhline(0.5, color="black", linewidth=0.7, linestyle="--")
    axes[1].set_ylabel("Pred up")
    axes[1].legend(loc="upper left")
    axes[2].plot(early["trade_date"], early["pred_big_up_prob"], label="pred_big_up_prob", color="#7c3aed", linewidth=1.2)
    axes[2].set_ylabel("Big up")
    axes[2].legend(loc="upper left")
    axes[3].plot(early["trade_date"], early["final_position"], label="final_position", color="#059669", linewidth=1.4)
    axes[3].set_ylim(0.78, 1.02)
    axes[3].set_ylabel("Position")
    axes[3].legend(loc="upper left")
    axes[4].bar(early["trade_date"], early["fwd_ret_1d"], label="fwd_ret_1d", color="#6b7280", width=1.0)
    axes[4].axhline(0, color="black", linewidth=0.7)
    axes[4].set_ylabel("Next-day ret")
    axes[4].legend(loc="upper left")
    axes[4].set_xlabel("Date")
    fig.suptitle("Early-Stress Signal Diagnostics")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "early_stress_signal_diagnostics.png", dpi=160)
    plt.close(fig)
