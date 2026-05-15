from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PLOT_DIR


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
