from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import TEST_END, TEST_START
from src.external_market_features import external_feature_columns, parse_trade_date


FINAL_COMPARISON_STRATEGIES = [
    "buy_hold",
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

EXTERNAL_STRATEGIES = [
    name for name in FINAL_COMPARISON_STRATEGIES if name.startswith("ml_external_")
]

HARD_EXTERNAL_STRATEGIES = [
    name
    for name in EXTERNAL_STRATEGIES
    if name.startswith("ml_external_risk_budget_enhancement")
    or name.startswith("ml_external_early_stress_risk_budget")
]

SOFT_CONFIRM_EXTERNAL_STRATEGIES = [
    name for name in EXTERNAL_STRATEGIES if name.startswith("ml_external_soft_confirm")
]

SOFT_LITE_EXTERNAL_STRATEGIES = [
    name for name in EXTERNAL_STRATEGIES if name.startswith("ml_external_soft_lite")
]

SOFT_EXTERNAL_STRATEGIES = SOFT_CONFIRM_EXTERNAL_STRATEGIES + SOFT_LITE_EXTERNAL_STRATEGIES

FINAL_COMPARISON_COLUMNS = [
    "strategy",
    "strategy_role",
    "total_return",
    "excess_return_vs_buy_hold",
    "max_drawdown",
    "drawdown_improvement_vs_buy_hold",
    "sharpe",
    "calmar",
    "average_position",
    "minimum_position",
    "maximum_position",
    "days_below_full_exposure",
    "total_turnover",
    "total_transaction_cost",
    "missed_upside",
    "avoided_downside",
    "avoided_minus_missed_upside",
    "net_timing_contribution",
    "benchmark_clone",
    "outperforms_buy_hold",
    "reduces_max_drawdown",
]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _strategy_role(strategy: str) -> str:
    if strategy == "buy_hold":
        return "benchmark"
    if strategy == "ml_risk_budget_enhancement":
        return "original_risk_budget"
    if strategy == "ml_early_stress_risk_budget":
        return "original_early_stress"
    if strategy.startswith("ml_external_risk_budget"):
        return "external_risk_budget_ablation"
    if strategy.startswith("ml_external_early_stress"):
        return "external_early_stress_ablation"
    if strategy.startswith("ml_external_soft_confirm_risk_budget"):
        return "external_soft_confirm_risk_budget"
    if strategy.startswith("ml_external_soft_confirm_early_stress"):
        return "external_soft_confirm_early_stress"
    if strategy.startswith("ml_external_soft_lite_risk_budget"):
        return "external_soft_lite_risk_budget"
    if strategy.startswith("ml_external_soft_lite_early_stress"):
        return "external_soft_lite_early_stress"
    return "other"


def _row(df: pd.DataFrame, strategy: str) -> pd.Series | None:
    rows = df.loc[df["strategy"] == strategy]
    return rows.iloc[0] if not rows.empty else None


def build_final_strategy_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    available = metrics.loc[metrics["strategy"].isin(FINAL_COMPARISON_STRATEGIES)].copy()
    if available.empty:
        return pd.DataFrame(columns=FINAL_COMPARISON_COLUMNS)

    order = {name: idx for idx, name in enumerate(FINAL_COMPARISON_STRATEGIES)}
    available["strategy_order"] = available["strategy"].map(order).fillna(999).astype(int)
    available = available.sort_values(["strategy_order", "strategy"]).reset_index(drop=True)
    buy_hold = _row(available, "buy_hold")
    benchmark_mdd = float(buy_hold["max_drawdown"]) if buy_hold is not None else np.nan
    benchmark_return = float(buy_hold["total_return"]) if buy_hold is not None else np.nan

    out = available.copy()
    out["strategy_role"] = out["strategy"].map(_strategy_role)
    if "benchmark_max_drawdown" in out.columns:
        baseline_mdd = out["benchmark_max_drawdown"].fillna(benchmark_mdd)
    else:
        baseline_mdd = benchmark_mdd
    out["drawdown_improvement_vs_buy_hold"] = out["max_drawdown"].astype(float) - baseline_mdd
    if "excess_return_vs_buy_hold" not in out.columns:
        out["excess_return_vs_buy_hold"] = out["total_return"].astype(float) - benchmark_return
    out["avoided_minus_missed_upside"] = out["avoided_downside"].astype(float) - out["missed_upside"].astype(float)
    out["outperforms_buy_hold"] = out["excess_return_vs_buy_hold"].astype(float) > 0
    out["reduces_max_drawdown"] = out["drawdown_improvement_vs_buy_hold"].astype(float) > 0
    cols = [col for col in FINAL_COMPARISON_COLUMNS if col in out.columns]
    return out.loc[:, cols]


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    use_cols = [col for col in columns if col in df.columns]
    if df.empty or not use_cols:
        return "No rows."
    lines = [
        "| " + " | ".join(use_cols) + " |",
        "| " + " | ".join(["---"] * len(use_cols)) + " |",
    ]
    for row in df[use_cols].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(_fmt(value, 6))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_final_strategy_comparison(metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    comparison = build_final_strategy_comparison(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "final_strategy_comparison.csv", index=False)
    display_cols = [
        "strategy",
        "total_return",
        "excess_return_vs_buy_hold",
        "max_drawdown",
        "drawdown_improvement_vs_buy_hold",
        "sharpe",
        "calmar",
        "average_position",
        "total_turnover",
        "missed_upside",
        "avoided_downside",
        "total_transaction_cost",
        "outperforms_buy_hold",
        "reduces_max_drawdown",
    ]
    text = f"""# Final Strategy Comparison

Test period: {TEST_START.date()} to {TEST_END.date()}

This table compares buy-and-hold, the original risk-budget/early-stress strategies, and the external-market ablation variants. All external variants are no-leverage and long-only.

{_markdown_table(comparison, display_cols)}
"""
    (output_dir / "final_strategy_comparison.md").write_text(text, encoding="utf-8")
    return comparison


def _raw_file_stats(raw_dir: Path, filename: str) -> dict[str, Any]:
    path = raw_dir / filename
    if not path.exists():
        return {"file": filename, "exists": False, "rows": 0, "columns": 0, "start": "NA", "end": "NA"}
    try:
        header = pd.read_csv(path, nrows=0)
        dates = pd.read_csv(path, usecols=["trade_date"], low_memory=False)
        parsed = parse_trade_date(dates["trade_date"]).dropna()
    except Exception:
        return {"file": filename, "exists": True, "rows": "NA", "columns": "NA", "start": "NA", "end": "NA"}
    return {
        "file": filename,
        "exists": True,
        "rows": int(len(dates)),
        "columns": int(len(header.columns)),
        "start": str(parsed.min().date()) if not parsed.empty else "NA",
        "end": str(parsed.max().date()) if not parsed.empty else "NA",
    }


def _best_external_rows(comparison: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    external = comparison.loc[comparison["strategy"].isin(EXTERNAL_STRATEGIES)].copy()
    if external.empty:
        return None, None, None
    best_return = external.sort_values("total_return", ascending=False).iloc[0]
    best_drawdown = external.sort_values("max_drawdown", ascending=False).iloc[0]
    best_sharpe = external.sort_values("sharpe", ascending=False, na_position="last").iloc[0]
    return best_return, best_drawdown, best_sharpe


def _best_rows_for(comparison: pd.DataFrame, strategies: list[str]) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    subset = comparison.loc[comparison["strategy"].isin(strategies)].copy()
    if subset.empty:
        return None, None, None
    best_return = subset.sort_values("total_return", ascending=False).iloc[0]
    best_drawdown = subset.sort_values("max_drawdown", ascending=False).iloc[0]
    best_sharpe = subset.sort_values("sharpe", ascending=False, na_position="last").iloc[0]
    return best_return, best_drawdown, best_sharpe


def _metric_delta(left: pd.Series | None, right: pd.Series | None, key: str) -> float | None:
    if left is None or right is None:
        return None
    try:
        return float(left[key]) - float(right[key])
    except (KeyError, TypeError, ValueError):
        return None


def build_external_summary_section(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "\n## External Market Data Enhancement\n- External comparison table was not generated.\n"
    buy_hold = _row(comparison, "buy_hold")
    risk_budget = _row(comparison, "ml_risk_budget_enhancement")
    early_stress = _row(comparison, "ml_early_stress_risk_budget")
    best_return, best_drawdown, best_sharpe = _best_external_rows(comparison)
    best_hard_return, best_hard_drawdown, _ = _best_rows_for(comparison, HARD_EXTERNAL_STRATEGIES)
    best_soft_confirm_return, best_soft_confirm_drawdown, best_soft_confirm_sharpe = _best_rows_for(
        comparison,
        SOFT_CONFIRM_EXTERNAL_STRATEGIES,
    )
    best_soft_lite_return, best_soft_lite_drawdown, best_soft_lite_sharpe = _best_rows_for(
        comparison,
        SOFT_LITE_EXTERNAL_STRATEGIES,
    )
    best_soft_return, best_soft_drawdown, best_soft_sharpe = _best_rows_for(comparison, SOFT_EXTERNAL_STRATEGIES)
    if best_return is None:
        return "\n## External Market Data Enhancement\n- No external ablation strategy rows were generated.\n"

    original_pool = [row for row in [risk_budget, early_stress] if row is not None]
    best_original = max(original_pool, key=lambda row: float(row["total_return"])) if original_pool else None
    external_beats_original = bool(best_original is not None and best_return["total_return"] > best_original["total_return"])
    external_reduces_drawdown_vs_buyhold = bool(best_drawdown is not None and best_drawdown["max_drawdown"] > buy_hold["max_drawdown"]) if buy_hold is not None else False
    return f"""
## External Market Data Enhancement
- External data used: index_daily, margin_daily, shibor_daily; all shifted by one raw market day before merge.
- Ablation variants: hard external index_only/index_margin/index_shibor/all, early_stress index_only/all, and soft-confirm/soft-lite all/index_only.
- Best external strategy by total return: `{best_return["strategy"]}`.
- Best external total return: {_fmt(best_return["total_return"])}; excess vs buy-and-hold: {_fmt(best_return["excess_return_vs_buy_hold"])}.
- Best external max drawdown strategy: `{best_drawdown["strategy"] if best_drawdown is not None else "NA"}`; max drawdown: {_fmt(best_drawdown["max_drawdown"] if best_drawdown is not None else None)}.
- Best external Sharpe strategy: `{best_sharpe["strategy"] if best_sharpe is not None else "NA"}`; Sharpe: {_fmt(best_sharpe["sharpe"] if best_sharpe is not None else None)}.
- External best-return strategy beats the best original risk strategy: {"Yes" if external_beats_original else "No"}.
- External variants reduce max drawdown vs buy-and-hold: {"Yes" if external_reduces_drawdown_vs_buyhold else "No"}.
- Tradeoff to inspect: external filters lower exposure when market-wide risk score rises, so they can reduce drawdown or volatility at the cost of missed upside in rebound periods.

## External Soft Confirmation Strategy
- Motivation: the hard external filter reduced drawdown but cut exposure on too many rebound/upside days, creating excessive missed upside.
- Soft design: external risk is only a confirmation layer. `watch` does not cut exposure; `high/extreme` risk receives only mild weights and needs internal stress confirmation for larger cuts.
- Best original soft-confirm strategy by total return: `{best_soft_confirm_return["strategy"] if best_soft_confirm_return is not None else "NA"}`.
- Best original soft-confirm total return: {_fmt(best_soft_confirm_return["total_return"] if best_soft_confirm_return is not None else None)}; excess vs buy-and-hold: {_fmt(best_soft_confirm_return["excess_return_vs_buy_hold"] if best_soft_confirm_return is not None else None)}.
- Best soft strategy overall by total return: `{best_soft_return["strategy"] if best_soft_return is not None else "NA"}`.
- Best soft total return overall: {_fmt(best_soft_return["total_return"] if best_soft_return is not None else None)}; excess vs buy-and-hold: {_fmt(best_soft_return["excess_return_vs_buy_hold"] if best_soft_return is not None else None)}.
- Best soft max drawdown strategy overall: `{best_soft_drawdown["strategy"] if best_soft_drawdown is not None else "NA"}`; max drawdown: {_fmt(best_soft_drawdown["max_drawdown"] if best_soft_drawdown is not None else None)}.
- Best soft Sharpe strategy overall: `{best_soft_sharpe["strategy"] if best_soft_sharpe is not None else "NA"}`; Sharpe: {_fmt(best_soft_sharpe["sharpe"] if best_soft_sharpe is not None else None)}; Calmar: {_fmt(best_soft_sharpe["calmar"] if best_soft_sharpe is not None else None)}.
- Soft best-return strategy beats buy-and-hold: {"Yes" if best_soft_return is not None and float(best_soft_return["excess_return_vs_buy_hold"]) > 0 else "No"}.
- Soft variants reduce max drawdown vs buy-and-hold: {"Yes" if best_soft_drawdown is not None and buy_hold is not None and float(best_soft_drawdown["max_drawdown"]) > float(buy_hold["max_drawdown"]) else "No"}.
- Soft best-return total-return delta vs original risk-budget: {_fmt(_metric_delta(best_soft_return, risk_budget, "total_return"))}.
- Soft best-drawdown improvement vs hard best-drawdown strategy: {_fmt(_metric_delta(best_soft_drawdown, best_hard_drawdown, "max_drawdown"))}.
- Missed-upside delta vs hard best-return strategy: {_fmt(_metric_delta(best_soft_return, best_hard_return, "missed_upside"))}; avoided-downside delta: {_fmt(_metric_delta(best_soft_return, best_hard_return, "avoided_downside"))}.
- Recommended main result: keep the original `ml_risk_budget_enhancement` if total return is the priority; report the best soft confirmation strategy as a supplementary drawdown-control experiment if it improves drawdown with materially less missed upside than the hard filter.

## External Soft-Lite Confirmation Strategy
- Soft-lite design: use the same external/internal confirmation logic, but with milder external weights and a tighter external overlay no-trade band so small risk discounts can be tested without hard de-risking.
- Best soft-lite strategy by total return: `{best_soft_lite_return["strategy"] if best_soft_lite_return is not None else "NA"}`.
- Best soft-lite total return: {_fmt(best_soft_lite_return["total_return"] if best_soft_lite_return is not None else None)}; excess vs buy-and-hold: {_fmt(best_soft_lite_return["excess_return_vs_buy_hold"] if best_soft_lite_return is not None else None)}.
- Best soft-lite max drawdown strategy: `{best_soft_lite_drawdown["strategy"] if best_soft_lite_drawdown is not None else "NA"}`; max drawdown: {_fmt(best_soft_lite_drawdown["max_drawdown"] if best_soft_lite_drawdown is not None else None)}.
- Best soft-lite Sharpe strategy: `{best_soft_lite_sharpe["strategy"] if best_soft_lite_sharpe is not None else "NA"}`; Sharpe: {_fmt(best_soft_lite_sharpe["sharpe"] if best_soft_lite_sharpe is not None else None)}; Calmar: {_fmt(best_soft_lite_sharpe["calmar"] if best_soft_lite_sharpe is not None else None)}.
- Soft-lite best-return strategy beats buy-and-hold: {"Yes" if best_soft_lite_return is not None and float(best_soft_lite_return["excess_return_vs_buy_hold"]) > 0 else "No"}.
- Soft-lite best-return total-return delta vs original soft-confirm best-return: {_fmt(_metric_delta(best_soft_lite_return, best_soft_confirm_return, "total_return"))}.
"""


def append_external_summary_section(summary_path: Path, comparison: pd.DataFrame) -> None:
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + build_external_summary_section(comparison),
        encoding="utf-8",
    )


def build_external_market_feature_report(
    feature_df: pd.DataFrame,
    comparison: pd.DataFrame,
    raw_dir: Path,
) -> str:
    external_cols = external_feature_columns(feature_df.columns)
    test = feature_df.loc[
        (pd.to_datetime(feature_df["date"]) >= pd.Timestamp(TEST_START))
        & (pd.to_datetime(feature_df["date"]) <= pd.Timestamp(TEST_END))
    ].copy()

    raw_stats = pd.DataFrame(
        [
            _raw_file_stats(raw_dir, "index_daily.csv"),
            _raw_file_stats(raw_dir, "margin_daily.csv"),
            _raw_file_stats(raw_dir, "shibor_daily.csv"),
            _raw_file_stats(raw_dir, "north_money.csv"),
            _raw_file_stats(raw_dir, "industry_daily.csv"),
        ]
    )

    score_cols = [
        "external_index_risk_score",
        "external_margin_risk_score",
        "external_shibor_risk_score",
        "external_market_risk_score",
    ]
    score_summary = test[score_cols].describe().T.reset_index().rename(columns={"index": "feature"}) if not test.empty else pd.DataFrame()
    condition_cols = [
        "external_condition_index_trend_weak",
        "external_condition_index_drawdown",
        "external_condition_index_volatility",
        "external_condition_index_liquidity",
        "external_condition_style_weak",
        "external_condition_margin_risk",
        "external_condition_shibor_tightening",
    ]
    condition_frequency = pd.DataFrame(
        {
            "condition": condition_cols,
            "test_period_frequency": [
                float(test[col].fillna(0.0).mean()) if col in test.columns and not test.empty else np.nan
                for col in condition_cols
            ],
        }
    )
    report_cols = [
        "strategy",
        "total_return",
        "excess_return_vs_buy_hold",
        "max_drawdown",
        "drawdown_improvement_vs_buy_hold",
        "sharpe",
        "calmar",
        "average_position",
        "missed_upside",
        "avoided_downside",
        "total_transaction_cost",
    ]
    comparison_view = comparison.loc[comparison["strategy"].isin(FINAL_COMPARISON_STRATEGIES)].copy()
    best_return, best_drawdown, best_sharpe = _best_external_rows(comparison)
    conclusion = build_external_summary_section(comparison).replace("\n## External Market Data Enhancement\n", "").strip()

    text = f"""# External Market Feature Report

## Raw Data Scope
{_markdown_table(raw_stats, ["file", "exists", "rows", "columns", "start", "end"])}

## Usage Decision
- Used in strategy features: `index_daily.csv`, `margin_daily.csv`, `shibor_daily.csv`.
- Excluded from the main strategy feature set: `north_money.csv` and `industry_daily.csv`.
- Reason: north-money data is not used as the main strategy feature under the current project constraint; `industry_daily.csv` is empty.
- Leakage control: every external feature is computed on the raw market timeline, shifted by one raw market day, then backward-aligned to the 880823 trading dates.

## Feature Construction
- External feature count in `feature_df`: {len(external_cols)}.
- Index features: multi-index returns, moving-average gaps, annualized volatility, drawdown, amount change/z-score, and style relative strength.
- Margin features: balance/buy change, rolling z-score, moving-average gap, and `margin_risk_off`.
- Shibor features: rate changes, z-scores, term slopes, and `shibor_tightening`.
- Composite risk score: 7 binary conditions covering index trend, index drawdown, index volatility, index liquidity, style weakness, margin risk, and Shibor tightening.

## Test-Period Risk Score Summary
{_markdown_table(score_summary, ["feature", "mean", "std", "min", "25%", "50%", "75%", "max"])}

## Test-Period Risk Condition Frequency
{_markdown_table(condition_frequency, ["condition", "test_period_frequency"])}

## Ablation Result Summary
{_markdown_table(comparison_view, report_cols)}

## Best External Variants
- Best by total return: `{best_return["strategy"] if best_return is not None else "NA"}` with total return {_fmt(best_return["total_return"] if best_return is not None else None)}.
- Best by max drawdown: `{best_drawdown["strategy"] if best_drawdown is not None else "NA"}` with max drawdown {_fmt(best_drawdown["max_drawdown"] if best_drawdown is not None else None)}.
- Best by Sharpe: `{best_sharpe["strategy"] if best_sharpe is not None else "NA"}` with Sharpe {_fmt(best_sharpe["sharpe"] if best_sharpe is not None else None)}.

## Conclusion
{conclusion}
"""
    return text


def write_external_market_feature_report(
    feature_df: pd.DataFrame,
    comparison: pd.DataFrame,
    raw_dir: Path,
    output_dir: Path,
) -> None:
    text = build_external_market_feature_report(feature_df, comparison, raw_dir)
    (output_dir / "external_market_feature_report.md").write_text(text, encoding="utf-8")
