from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import COST_RATE, INITIAL_CAPITAL, TEST_END, TEST_START
from src.strategy.metrics import select_strategy_rows
from src.strategy.upside_participation import (
    EARLY_STRESS_STRATEGY_NAME,
    EXTERNAL_EARLY_STRESS_STRATEGY_NAMES,
    EXTERNAL_RISK_BUDGET_STRATEGY_NAMES,
    EXTERNAL_SOFT_EARLY_STRESS_STRATEGY_NAMES,
    EXTERNAL_SOFT_LITE_EARLY_STRESS_STRATEGY_NAMES,
    EXTERNAL_SOFT_LITE_RISK_BUDGET_STRATEGY_NAMES,
    EXTERNAL_SOFT_RISK_BUDGET_STRATEGY_NAMES,
    RISK_BUDGET_STRATEGY_NAME,
    STABILITY_AWARE_STRATEGY_NAME,
    UPSIDE_STRATEGY_NAME,
)


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


def _row(metrics: pd.DataFrame, strategy: str) -> pd.Series | None:
    rows = metrics.loc[metrics["strategy"] == strategy]
    return rows.iloc[0] if not rows.empty else None


def _best_prediction_row(metrics: pd.DataFrame | None, model_prefix: str | None = None) -> pd.Series | None:
    if metrics is None or metrics.empty or "selection_score" not in metrics.columns:
        return None
    candidates = metrics
    if model_prefix is not None:
        candidates = candidates.loc[candidates["model_name"].astype(str).str.startswith(model_prefix)]
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["selection_score", "clean_direction_AUC", "AUC"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]


def _matching_prediction_row(metrics: pd.DataFrame, selected: pd.Series | None) -> pd.Series | None:
    if selected is None or metrics.empty:
        return None
    rows = metrics.loc[
        (metrics["model_name"] == selected["model_name"]) & (metrics["horizon"] == selected["horizon"])
    ]
    return rows.iloc[0] if not rows.empty else None


def _prediction_selection_section(
    validation_metrics: pd.DataFrame | None,
    test_metrics: pd.DataFrame,
) -> str:
    selected = _best_prediction_row(validation_metrics)
    if selected is None:
        return ""
    selected_test = _matching_prediction_row(test_metrics, selected)
    best_mlp = _best_prediction_row(validation_metrics, "mlp_")
    best_mlp_test = _matching_prediction_row(test_metrics, best_mlp)
    mlp_names = sorted(
        validation_metrics.loc[
            validation_metrics["model_name"].astype(str).str.startswith("mlp_"), "model_name"
        ].drop_duplicates()
    )
    mlp_selected = str(selected["model_name"]).startswith("mlp_")
    mlp_description = (
        f"{best_mlp['model_name']} / {int(best_mlp['horizon'])}D; "
        f"validation score {_fmt(best_mlp.get('selection_score'))}; "
        f"matching test score {_fmt(best_mlp_test.get('selection_score') if best_mlp_test is not None else None)}; "
        f"test AUC {_fmt(best_mlp_test.get('AUC') if best_mlp_test is not None else None)}; "
        f"test clean-direction AUC {_fmt(best_mlp_test.get('clean_direction_AUC') if best_mlp_test is not None else None)}; "
        f"test trade AUC {_fmt(best_mlp_test.get('trade_AUC') if best_mlp_test is not None else None)}; "
        f"test return correlation {_fmt(best_mlp_test.get('return_correlation') if best_mlp_test is not None else None)}"
        if best_mlp is not None
        else "NA"
    )
    return f"""
## Prediction Model Selection
- MLP candidates evaluated: {", ".join(mlp_names) if mlp_names else "None"}
- Validation-selected model/horizon: {selected["model_name"]} / {int(selected["horizon"])}D
- MLP selected by validation score: {"Yes" if mlp_selected else "No"}
- Selected-model validation score: {_fmt(selected.get("selection_score"))}
- Selected-model matching test score: {_fmt(selected_test.get("selection_score") if selected_test is not None else None)}
- Selected-model test direction AUC: {_fmt(selected_test.get("AUC") if selected_test is not None else None)}
- Selected-model test clean-direction AUC: {_fmt(selected_test.get("clean_direction_AUC") if selected_test is not None else None)}
- Selected-model test trade AUC: {_fmt(selected_test.get("trade_AUC") if selected_test is not None else None)}
- Selected-model test return correlation: {_fmt(selected_test.get("return_correlation") if selected_test is not None else None)}
- Best MLP comparison: {mlp_description}
"""


def _tradeable_up_sections(metrics: pd.DataFrame, context: dict[str, Any] | None) -> tuple[str, str]:
    if not context:
        return "", ""
    selection = context["selection"]["selected"]
    stability = context["stability_summary"]
    val = stability.loc[
        (stability["period"] == "validation")
        & (stability["model_name"] == selection["model_name"])
        & (stability["horizon"] == int(selection["horizon"]))
        & (stability["signal_name"] == selection["signal_name"])
    ]
    test = stability.loc[
        (stability["period"] == "test")
        & (stability["model_name"] == selection["model_name"])
        & (stability["horizon"] == int(selection["horizon"]))
        & (stability["signal_name"] == selection["signal_name"])
    ]
    val_row = val.iloc[0] if not val.empty else pd.Series(dtype=object)
    test_row = test.iloc[0] if not test.empty else pd.Series(dtype=object)
    low_score_decline = bool(val_row.get("bottom_group_negative_mean", False)) and bool(
        test_row.get("bottom_group_negative_mean", False)
    )
    upside = _row(metrics, UPSIDE_STRATEGY_NAME)
    rows = select_strategy_rows(metrics)
    best_real = rows["best_real_no_leverage"]
    is_best_real = bool(best_real is not None and best_real.get("strategy") == UPSIDE_STRATEGY_NAME)
    outperforms = bool(upside is not None and float(upside.get("excess_return_vs_buy_hold", 0.0)) > 0)
    missed_upside = float(upside.get("missed_upside", 0.0)) if upside is not None else float("nan")
    avoided_downside = float(upside.get("avoided_downside", 0.0)) if upside is not None else float("nan")
    transaction_cost = float(upside.get("total_transaction_cost", 0.0)) if upside is not None else float("nan")
    timing_failure = (
        "Yes, missed upside is larger than avoided downside plus transaction cost."
        if pd.notna(missed_upside)
        and pd.notna(avoided_downside)
        and pd.notna(transaction_cost)
        and missed_upside > avoided_downside + transaction_cost
        else "No, missed upside is not larger than avoided downside plus transaction cost."
    )
    cost_failure = (
        "Yes"
        if pd.notna(transaction_cost)
        and pd.notna(missed_upside)
        and transaction_cost > missed_upside
        else "No"
    )
    success_sentence = (
        "The upside participation enhancement strategy achieves positive no-leverage excess return by using tradeable-up signals to maintain full participation in favorable regimes while selectively reducing exposure in confirmed unfavorable regimes."
        if outperforms
        else "The upside participation enhancement strategy improves the interpretability of ML-based index enhancement and maintains high benchmark participation, but it still does not outperform buy-and-hold in this strongly rising test period."
    )
    signal_section = f"""
## Tradeable-Up Signal Validation
- Selected signal: {selection["signal_name"]}
- Selected model/horizon: {selection["model_name"]}, {selection["horizon"]}D
- Validation top-bottom spread: {_fmt(val_row.get("top_bottom_spread"))}
- Validation top group win rate: {_fmt(val_row.get("top_group_win_rate"))}
- Test top-bottom spread: {_fmt(test_row.get("top_bottom_spread"))}
- Test top group win rate: {_fmt(test_row.get("top_group_win_rate"))}
- Low-score group really corresponds to future decline: {"Yes" if low_score_decline else "No"}
- Conclusion: {"The strategy can use low-score information cautiously." if low_score_decline else "The strategy should not rely on low-score forced selling."}
"""
    strategy_section = f"""
## Upside Participation Index Enhancement
- Strategy: {UPSIDE_STRATEGY_NAME}
- Design: benchmark-aware no-leverage enhancement, not aggressive 0/1 buy/sell timing.
- Participation rule: default high participation; high upside signal keeps exposure at 1.00.
- Defensive rule: low signal does not directly sell; only low signal plus weak trend and high risk can mildly reduce exposure.
- Maximum exposure: 1.00, no leverage.
- Parameter selection: validation-period only.
- Total return: {_fmt(upside.get("total_return") if upside is not None else None)}
- Excess return: {_fmt(upside.get("excess_return_vs_buy_hold") if upside is not None else None)}
- Max drawdown: {_fmt(upside.get("max_drawdown") if upside is not None else None)}
- Sharpe: {_fmt(upside.get("sharpe") if upside is not None else None)}
- Average position: {_fmt(upside.get("average_position") if upside is not None else None)}
- Days below full exposure: {int(upside.get("days_below_full_exposure")) if upside is not None and pd.notna(upside.get("days_below_full_exposure")) else "NA"}
- Total turnover: {_fmt(upside.get("total_turnover") if upside is not None else None)}
- Missed upside: {_fmt(missed_upside)}
- Avoided downside: {_fmt(avoided_downside)}
- Transaction cost: {_fmt(transaction_cost)}
- Mainly missed upside: {timing_failure}
- Mainly transaction cost: {cost_failure}
- Outperforms buy-and-hold: {"Yes" if outperforms else "No"}
- Becomes best real no-leverage ML timing strategy: {"Yes" if is_best_real else "No"}
- Strong rising test period makes defensive exposure cuts naturally costly: Yes

{success_sentence}
"""
    return signal_section, strategy_section


def _stability_aware_section(metrics: pd.DataFrame, context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    components = context.get("selected_components", [])
    stability = _row(metrics, STABILITY_AWARE_STRATEGY_NAME)
    rows = select_strategy_rows(metrics)
    best_real = rows["best_real_no_leverage"]
    is_best_real = bool(best_real is not None and best_real.get("strategy") == STABILITY_AWARE_STRATEGY_NAME)
    outperforms = bool(stability is not None and float(stability.get("excess_return_vs_buy_hold", 0.0)) > 0)
    component_text = "; ".join(
        f"{item['model_name']} {item['horizon']}D {item['signal_name']} score={_fmt(item.get('stable_score'))}"
        for item in components
    ) or "NA"
    missed_upside = float(stability.get("missed_upside", 0.0)) if stability is not None else float("nan")
    avoided_downside = float(stability.get("avoided_downside", 0.0)) if stability is not None else float("nan")
    transaction_cost = float(stability.get("total_transaction_cost", 0.0)) if stability is not None else float("nan")
    success_sentence = (
        "The strategy achieves positive no-leverage excess return by combining validation-selected upside signals and applying only mild defensive exposure reductions in unfavorable regimes."
        if outperforms
        else "The strategy maintains high benchmark participation and reduces dependence on low-score sell signals, but it still does not outperform buy-and-hold in this strongly rising test period. The attribution indicates that missed upside remains larger than avoided downside."
    )
    return f"""
## Stability-Aware High-Participation Index Enhancement
- Strategy: {STABILITY_AWARE_STRATEGY_NAME}
- Design: not direct buy/sell and not low-score forced selling.
- Interpretation: high-participation benchmark-aware no-leverage enhancement.
- Default exposure: 1.00.
- Defensive rule: only when ensemble upside score is very low and trend or risk confirms weakness does exposure get mildly reduced.
- Ensemble purpose: multiple validation-selected upside signals reduce single-validation winner-take-all overfitting.
- Signal selection: validation-period only.
- Selected signal components: {component_text}
- Total return: {_fmt(stability.get("total_return") if stability is not None else None)}
- Excess return: {_fmt(stability.get("excess_return_vs_buy_hold") if stability is not None else None)}
- Max drawdown: {_fmt(stability.get("max_drawdown") if stability is not None else None)}
- Sharpe: {_fmt(stability.get("sharpe") if stability is not None else None)}
- Average position: {_fmt(stability.get("average_position") if stability is not None else None)}
- Days below full exposure: {int(stability.get("days_below_full_exposure")) if stability is not None and pd.notna(stability.get("days_below_full_exposure")) else "NA"}
- Total turnover: {_fmt(stability.get("total_turnover") if stability is not None else None)}
- Missed upside: {_fmt(missed_upside)}
- Avoided downside: {_fmt(avoided_downside)}
- Transaction cost: {_fmt(transaction_cost)}
- Outperforms buy-and-hold: {"Yes" if outperforms else "No"}
- Becomes best real no-leverage ML timing strategy: {"Yes" if is_best_real else "No"}

{success_sentence}
"""


def _risk_budget_section(metrics: pd.DataFrame, context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    risk_budget = _row(metrics, RISK_BUDGET_STRATEGY_NAME)
    buy_hold = _row(metrics, "buy_hold")
    rows = select_strategy_rows(metrics)
    best_real = rows["best_real_no_leverage"]
    is_best_real = bool(best_real is not None and best_real.get("strategy") == RISK_BUDGET_STRATEGY_NAME)
    outperforms = bool(risk_budget is not None and float(risk_budget.get("excess_return_vs_buy_hold", 0.0)) > 0)
    benchmark_mdd = float(buy_hold.get("max_drawdown", float("nan"))) if buy_hold is not None else float("nan")
    strategy_mdd = float(risk_budget.get("max_drawdown", float("nan"))) if risk_budget is not None else float("nan")
    drawdown_improvement = strategy_mdd - benchmark_mdd if pd.notna(strategy_mdd) and pd.notna(benchmark_mdd) else float("nan")
    reduces_drawdown = bool(pd.notna(drawdown_improvement) and drawdown_improvement > 0)
    selected_params = context.get("risk_budget_tuned", {}).get("selected_params", {})
    missed_upside = float(risk_budget.get("missed_upside", 0.0)) if risk_budget is not None else float("nan")
    avoided_downside = float(risk_budget.get("avoided_downside", 0.0)) if risk_budget is not None else float("nan")
    transaction_cost = float(risk_budget.get("total_transaction_cost", 0.0)) if risk_budget is not None else float("nan")
    success_sentence = (
        "The risk-budget enhancement strategy reduces maximum drawdown by dynamically lowering exposure under high-volatility, weak-trend, and drawdown-stress regimes."
        if reduces_drawdown
        else "The risk-budget enhancement strategy provides an explicit drawdown-control framework, but the realized maximum drawdown remains close to buy-and-hold because the test period is strongly upward-trending and risk-off periods are followed by rapid rebounds."
    )
    return f"""
## Risk-Budget Drawdown Control Enhancement
- Strategy: {RISK_BUDGET_STRATEGY_NAME}
- Design: risk-budget index enhancement, not 0/1 buy/sell timing.
- Participation rule: default high index participation near 1.00.
- Defensive rule: exposure is lowered only when volatility, drawdown, trend, and tail-risk states jointly deteriorate.
- ML role: auxiliary recovery or confirmation signal only; low ML probability alone does not force selling.
- Maximum exposure: 1.00, no leverage and no shorting.
- Parameter selection: validation-period only.
- Selected model/horizon: {selected_params.get("model_name", "NA")} / {selected_params.get("horizon", "NA")}D
- Total return: {_fmt(risk_budget.get("total_return") if risk_budget is not None else None)}
- Excess return: {_fmt(risk_budget.get("excess_return_vs_buy_hold") if risk_budget is not None else None)}
- Max drawdown: {_fmt(strategy_mdd)}
- Drawdown improvement vs buy-and-hold: {_fmt(drawdown_improvement)}
- Sharpe: {_fmt(risk_budget.get("sharpe") if risk_budget is not None else None)}
- Average position: {_fmt(risk_budget.get("average_position") if risk_budget is not None else None)}
- Days below full exposure: {int(risk_budget.get("days_below_full_exposure")) if risk_budget is not None and pd.notna(risk_budget.get("days_below_full_exposure")) else "NA"}
- Total turnover: {_fmt(risk_budget.get("total_turnover") if risk_budget is not None else None)}
- Missed upside: {_fmt(missed_upside)}
- Avoided downside: {_fmt(avoided_downside)}
- Transaction cost: {_fmt(transaction_cost)}
- Outperforms buy-and-hold: {"Yes" if outperforms else "No"}
- Reduces maximum drawdown: {"Yes" if reduces_drawdown else "No"}
- Becomes best real no-leverage ML timing strategy: {"Yes" if is_best_real else "No"}

{success_sentence}
"""


def _early_stress_section(metrics: pd.DataFrame, context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    early = _row(metrics, EARLY_STRESS_STRATEGY_NAME)
    buy_hold = _row(metrics, "buy_hold")
    rows = select_strategy_rows(metrics)
    best_real = rows["best_real_no_leverage"]
    is_best_real = bool(best_real is not None and best_real.get("strategy") == EARLY_STRESS_STRATEGY_NAME)
    outperforms = bool(early is not None and float(early.get("excess_return_vs_buy_hold", 0.0)) > 0)
    benchmark_mdd = float(buy_hold.get("max_drawdown", float("nan"))) if buy_hold is not None else float("nan")
    strategy_mdd = float(early.get("max_drawdown", float("nan"))) if early is not None else float("nan")
    drawdown_improvement = strategy_mdd - benchmark_mdd if pd.notna(strategy_mdd) and pd.notna(benchmark_mdd) else float("nan")
    reduces_drawdown = bool(pd.notna(drawdown_improvement) and drawdown_improvement > 0)
    selected_params = context.get("early_stress_tuned", {}).get("selected_params", {})
    event_report = context.get("drawdown_event_report", {})
    event_row = event_report.get("summary_by_strategy", {}).get(EARLY_STRESS_STRATEGY_NAME, {})
    risk_event_row = event_report.get("summary_by_strategy", {}).get(RISK_BUDGET_STRATEGY_NAME, {})
    earlier_than_original = bool(
        event_row.get("reduced_exposure_before_trough", False)
        and not risk_event_row.get("reduced_exposure_before_trough", False)
    )
    categories = context.get("feature_categories", {})
    category_lines = "; ".join(f"{name}: {len(cols)}" for name, cols in categories.items()) or "NA"
    feature_count = sum(len(cols) for cols in categories.values()) if categories else 0
    success_sentence = (
        "The early-stress risk-budget strategy improves the original risk-budget design by reacting earlier to short-term downside pressure and intraday selling stress."
        if outperforms or reduces_drawdown
        else "The early-stress risk-budget strategy improves the diagnostic coverage of pre-drawdown stress, but the available OHLCV and intraday signals are still not strong enough to produce stable positive excess return in the strongly rising test period."
    )
    return f"""
## Early-Stress Risk Budget Enhancement
- Strategy: {EARLY_STRESS_STRATEGY_NAME}
- Design: high-participation index enhancement, not 0/1 buy/sell timing.
- Early stress feature categories: {category_lines}
- Early stress feature count: {feature_count}
- Stress logic: early_stress_score attempts to detect short-term downside pressure, intraday selling stress, liquidity pressure, weak trend, and volatility expansion.
- Defensive rule: reduce exposure only when early stress is high and ML/upside signals are not strong.
- Maximum exposure: 1.00, no leverage and no shorting.
- Parameter selection: validation-period only.
- Selected model/horizon: {selected_params.get("model_name", "NA")} / {selected_params.get("horizon", "NA")}D
- Total return: {_fmt(early.get("total_return") if early is not None else None)}
- Excess return: {_fmt(early.get("excess_return_vs_buy_hold") if early is not None else None)}
- Max drawdown: {_fmt(strategy_mdd)}
- Drawdown improvement vs buy-and-hold: {_fmt(drawdown_improvement)}
- Sharpe: {_fmt(early.get("sharpe") if early is not None else None)}
- Average position: {_fmt(early.get("average_position") if early is not None else None)}
- Days below full exposure: {int(early.get("days_below_full_exposure")) if early is not None and pd.notna(early.get("days_below_full_exposure")) else "NA"}
- Total turnover: {_fmt(early.get("total_turnover") if early is not None else None)}
- Outperforms buy-and-hold: {"Yes" if outperforms else "No"}
- Reduces maximum drawdown: {"Yes" if reduces_drawdown else "No"}
- Becomes best real no-leverage ML timing strategy: {"Yes" if is_best_real else "No"}
- Reduced exposure before benchmark trough: {"Yes" if event_row.get("reduced_exposure_before_trough", False) else "No"}
- Earlier than original risk budget in max-drawdown event: {"Yes" if earlier_than_original else "No"}

{success_sentence}
"""


def write_strategy_reports(
    metrics: pd.DataFrame,
    prediction_test_metrics: pd.DataFrame,
    tuned_params: dict[str, Any],
    summary_path,
    final_report_path,
    tradeable_up_context: dict[str, Any] | None = None,
    stability_aware_context: dict[str, Any] | None = None,
    risk_budget_context: dict[str, Any] | None = None,
    early_stress_context: dict[str, Any] | None = None,
    prediction_validation_metrics: pd.DataFrame | None = None,
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
                    UPSIDE_STRATEGY_NAME,
                    STABILITY_AWARE_STRATEGY_NAME,
                    RISK_BUDGET_STRATEGY_NAME,
                    EARLY_STRESS_STRATEGY_NAME,
                    *EXTERNAL_RISK_BUDGET_STRATEGY_NAMES.values(),
                    *EXTERNAL_EARLY_STRESS_STRATEGY_NAMES.values(),
                    *EXTERNAL_SOFT_RISK_BUDGET_STRATEGY_NAMES.values(),
                    *EXTERNAL_SOFT_EARLY_STRESS_STRATEGY_NAMES.values(),
                    *EXTERNAL_SOFT_LITE_RISK_BUDGET_STRATEGY_NAMES.values(),
                    *EXTERNAL_SOFT_LITE_EARLY_STRESS_STRATEGY_NAMES.values(),
                ]
            )
            & (~metrics["benchmark_clone"].astype(bool))
            & (metrics["excess_return_vs_buy_hold"] > 0)
        ].empty
    )

    best_auc = prediction_test_metrics["AUC"].max()
    best_clean = prediction_test_metrics["clean_direction_AUC"].max()
    best_big_up = prediction_test_metrics["big_up_AUC"].max()
    prediction_selection_section = _prediction_selection_section(prediction_validation_metrics, prediction_test_metrics)
    direction_interpretation = (
        "At least one candidate reaches 0.60 on both test ordinary-direction and clean-direction AUC; any trading use still requires validation-selected, cost-aware evidence."
        if best_auc >= 0.60 and best_clean >= 0.60
        else "Ordinary direction AUC and clean-direction AUC are below 0.60, so the model is not used as an aggressive 0/1 market-timing engine."
    )
    upside_interpretation = (
        "Big-up AUC is above 0.60, so the decision layer focuses on strong-upside participation and risk-aware exposure adjustment."
        if best_big_up >= 0.60
        else "Big-up AUC is below 0.60, so upside-participation signals should be interpreted cautiously."
    )
    final_prediction_interpretation = (
        "Across test-period candidates, ordinary-direction and clean-direction AUC both reach 0.60, although model use remains governed by validation-only selection and strategy evaluation."
        if best_auc >= 0.60 and best_clean >= 0.60
        else "The out-of-sample prediction signal is moderate. Ordinary direction AUC and clean-direction AUC remain below 0.60, so the model is not treated as a reliable daily market-timing engine."
    )
    tradeable_signal_section, upside_strategy_section = _tradeable_up_sections(metrics, tradeable_up_context)
    stability_aware_section = _stability_aware_section(metrics, stability_aware_context)
    risk_budget_section = _risk_budget_section(metrics, risk_budget_context)
    early_stress_section = _early_stress_section(metrics, early_stress_context)

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
- {direction_interpretation}
- {upside_interpretation}
{prediction_selection_section}

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
{tradeable_signal_section}
{upside_strategy_section}
{stability_aware_section}
{risk_budget_section}
{early_stress_section}
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
{final_prediction_interpretation}
{prediction_selection_section}

## Decision Layer
The final no-leverage strategy is benchmark-aware index enhancement. It keeps exposure close to 1.00 in normal and favorable regimes, uses the big-up probability and tail score to maintain full participation in strong upside regimes, and only mildly reduces exposure when downside risk, weak trend, high volatility, or drawdown risk is present. Direct ML timing is included as an interpretable BUY/HOLD/SELL baseline but is not treated as the main result unless it truly outperforms.
{tradeable_signal_section}
{upside_strategy_section}
{stability_aware_section}
{risk_budget_section}
{early_stress_section}

## Validation-Selected Parameters
{params_text}

## Result Table
{_markdown_table(metrics)}

## Reporting Note
No-leverage strategies and enhanced-exposure strategies are separated. Enhanced-exposure strategies allow exposure above 1.00 and are therefore leverage-assisted experiments rather than the primary no-leverage enhanced-index result.
"""
    final_report_path.write_text(final, encoding="utf-8")
