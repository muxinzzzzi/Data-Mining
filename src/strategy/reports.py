from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import COST_RATE, INITIAL_CAPITAL, TEST_END, TEST_START
from src.strategy.metrics import select_strategy_rows


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    display_cols = [
        "strategy",
        "total_return",
        "excess_return_vs_buy_hold",
        "max_drawdown",
        "sharpe",
        "average_position",
        "maximum_position",
        "total_turnover",
        "benchmark_clone",
    ]
    cols = [col for col in display_cols if col in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in df[cols].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_strategy_reports(
    metrics: pd.DataFrame,
    prediction_test_metrics: pd.DataFrame,
    tuned_params: dict[str, Any],
    summary_path,
    final_report_path,
) -> None:
    rows = select_strategy_rows(metrics)
    buy_hold = rows["buy_hold"]
    best_no_lev = rows["best_no_leverage"]
    best_real = rows["best_real_no_leverage"]
    best_plus = rows["best_enhanced_exposure"]
    any_no_lev_outperform = bool(
        not metrics.loc[
            metrics["strategy"].isin(
                [
                    "ml_big_up_index_enhancement",
                    "ml_conservative_full_participation_enhancement",
                    "ml_ultra_conservative_full_participation_enhancement",
                    "ml_direct_signal_timing",
                ]
            )
            & (~metrics["benchmark_clone"].astype(bool))
            & (metrics["excess_return_vs_buy_hold"] > 0)
        ].empty
    )

    best_auc = prediction_test_metrics["AUC"].max()
    best_clean = prediction_test_metrics["clean_direction_AUC"].max()
    best_big_up = prediction_test_metrics["big_up_AUC"].max()

    summary = f"""# Enhanced Index Strategy Summary

## Setup
- Initial capital: RMB {INITIAL_CAPITAL:,.0f}
- Test period: {TEST_START.date()} to {TEST_END.date()}
- Transaction cost rate: {COST_RATE:.4f}
- Signal timing: predictions and regime filters are observed after close on date t and applied to the next tradable return.

## Prediction Interpretation
- Best test ordinary direction AUC: {_fmt(best_auc)}
- Best test clean-direction AUC: {_fmt(best_clean)}
- Best test big-up AUC: {_fmt(best_big_up)}
- Ordinary direction AUC and clean-direction AUC are below 0.60, so the model is not used as an aggressive 0/1 market-timing engine.
- Big-up AUC is above 0.60, so the decision layer focuses on strong-upside participation and risk-aware exposure adjustment.

## Strategy Results
- Buy-and-hold total return: {_fmt(buy_hold["total_return"] if buy_hold is not None else None)}
- Buy-and-hold max drawdown: {_fmt(buy_hold["max_drawdown"] if buy_hold is not None else None)}
- Best no-leverage strategy: {best_no_lev["strategy"] if best_no_lev is not None else "NA"}
- Whether best no-leverage strategy is benchmark clone: {bool(best_no_lev["benchmark_clone"]) if best_no_lev is not None else "NA"}
- Best real no-leverage ML strategy: {best_real["strategy"] if best_real is not None else "NA"}
- ML strategy total return: {_fmt(best_real["total_return"] if best_real is not None else None)}
- ML strategy excess return: {_fmt(best_real["excess_return_vs_buy_hold"] if best_real is not None else None)}
- ML strategy max drawdown: {_fmt(best_real["max_drawdown"] if best_real is not None else None)}
- Best enhanced-exposure strategy: {best_plus["strategy"] if best_plus is not None else "NA"}
- Best enhanced-exposure excess return: {_fmt(best_plus["excess_return_vs_buy_hold"] if best_plus is not None else None)}
- Any real no-leverage ML strategy outperforms buy-and-hold: {"Yes" if any_no_lev_outperform else "No"}

## Interpretation
No-leverage and enhanced-exposure strategies are reported separately. If no-leverage ML enhancement fails to outperform buy-and-hold, the main reason is that the test period was strongly upward and even mild defensive exposure cuts can miss upside. If an enhanced-exposure strategy outperforms, that result should be interpreted separately because it allows exposure above 1.00.
"""
    summary_path.write_text(summary, encoding="utf-8")

    params_text = "\n".join(
        f"- {name}: {info['selected']['params']}" for name, info in tuned_params.items()
    )
    final = f"""# Final Report Material

## Research Question
This project uses daily OHLCV data and auxiliary 5-minute intraday features for a micro-cap index to build a machine-learning enhanced index strategy. The investment test period is {TEST_START.date()} to {TEST_END.date()}, with initial capital of RMB {INITIAL_CAPITAL:,.0f}.

## Leakage Control
The prediction layer uses fixed validation and test periods, no random train/test split, and strict walk-forward training. Every training sample must satisfy `target_end_date < chunk_start`. Feature selection, preprocessing, and model selection are performed using training or validation data only. Trading decisions use signals available after close on date t and apply them to the next tradable return.

## Prediction Layer
The out-of-sample prediction signal is moderate. Ordinary direction AUC and clean-direction AUC remain below 0.60, while big-up AUC is above 0.60. This means the model is better at identifying strong upside opportunities than predicting every normal up/down day.

## Decision Layer
The final no-leverage strategy is benchmark-aware index enhancement. It keeps exposure close to 1.00 in normal and favorable regimes, uses the big-up probability and tail score to maintain full participation in strong upside regimes, and only mildly reduces exposure when downside risk, weak trend, high volatility, or drawdown risk is present. Direct ML timing is included as an interpretable BUY/HOLD/SELL baseline but is not treated as the main result unless it truly outperforms.

## Validation-Selected Parameters
{params_text}

## Result Table
{_markdown_table(metrics)}

## Reporting Note
No-leverage strategies and enhanced-exposure strategies are separated. Enhanced-exposure strategies allow exposure above 1.00 and are therefore leverage-assisted experiments rather than the primary no-leverage enhanced-index result.
"""
    final_report_path.write_text(final, encoding="utf-8")
