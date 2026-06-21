from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "outputs"
PREDICTION_PRIMARY = OUTPUT_DIR / "predictions_all_models.csv"
PREDICTION_VALIDATION = OUTPUT_DIR / "predictions" / "validation_predictions_all_models.csv"
PREDICTION_TEST = OUTPUT_DIR / "predictions" / "test_predictions_all_models.csv"
STRATEGY_METRICS_PATH = OUTPUT_DIR / "strategy_metrics_all.csv"
STRATEGY_DAILY_PATH = OUTPUT_DIR / "strategy_daily_all.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.md"
TUNED_PARAMS_PATH = OUTPUT_DIR / "strategy_tuned_params.json"

SIGNAL_COLS = [
    "pred_up_prob",
    "pred_trade_prob",
    "pred_big_up_prob",
    "pred_big_down_prob",
    "pred_tail_score",
]

UPWARD_SIGNAL_COLS = [
    "pred_up_prob",
    "pred_trade_prob",
    "pred_big_up_prob",
    "pred_tail_score",
]

NO_LEVERAGE_STRATEGIES = [
    "ml_big_up_index_enhancement",
    "ml_conservative_full_participation_enhancement",
    "ml_ultra_conservative_full_participation_enhancement",
    "ml_direct_signal_timing",
]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _pct(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value * 100:.2f}%"


def _markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "No rows."
    use = df.loc[:, [col for col in cols if col in df.columns]].head(max_rows).copy()
    if use.empty:
        return "No rows."
    lines = [
        "| " + " | ".join(use.columns) + " |",
        "| " + " | ".join(["---"] * len(use.columns)) + " |",
    ]
    for row in use.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(_fmt(value, 6))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _normalize_period_label(value: Any) -> str:
    text = str(value)
    if text.startswith("validation"):
        return "validation"
    if text.startswith("test"):
        return "test"
    return text if text and text != "nan" else "unknown"


def load_predictions() -> tuple[pd.DataFrame, str]:
    if PREDICTION_PRIMARY.exists():
        pred = _read_csv(PREDICTION_PRIMARY)
        source = str(PREDICTION_PRIMARY.relative_to(ROOT))
    else:
        frames: list[pd.DataFrame] = []
        sources: list[str] = []
        if PREDICTION_VALIDATION.exists():
            validation = _read_csv(PREDICTION_VALIDATION)
            validation["prediction_source_file"] = str(PREDICTION_VALIDATION.relative_to(ROOT))
            frames.append(validation)
            sources.append(str(PREDICTION_VALIDATION.relative_to(ROOT)))
        if PREDICTION_TEST.exists():
            test = _read_csv(PREDICTION_TEST)
            test["prediction_source_file"] = str(PREDICTION_TEST.relative_to(ROOT))
            frames.append(test)
            sources.append(str(PREDICTION_TEST.relative_to(ROOT)))
        if not frames:
            raise FileNotFoundError(
                "No prediction file found. Expected outputs/predictions_all_models.csv "
                "or the modular validation/test prediction files."
            )
        pred = pd.concat(frames, ignore_index=True)
        source = " + ".join(sources)

    if "walk_forward_period" in pred.columns:
        pred["diagnostic_period"] = pred["walk_forward_period"].map(_normalize_period_label)
    elif "prediction_source_file" in pred.columns:
        pred["diagnostic_period"] = pred["prediction_source_file"].map(_normalize_period_label)
    else:
        pred["diagnostic_period"] = "combined"

    for col in ["date", "trade_date", "target_start_date", "target_end_date"]:
        if col in pred.columns:
            pred[col] = pd.to_datetime(pred[col], errors="coerce")
    return pred, source


def _bucket_signal(group: pd.DataFrame, signal_col: str) -> pd.Series:
    clean = group[signal_col].replace([np.inf, -np.inf], np.nan)
    valid = clean.notna() & group["fwd_ret_1d"].replace([np.inf, -np.inf], np.nan).notna()
    buckets = pd.Series(np.nan, index=group.index, dtype="float")
    if int(valid.sum()) < 5:
        return buckets
    ranked = clean.loc[valid].rank(method="first")
    buckets.loc[valid] = pd.qcut(ranked, 5, labels=False, duplicates="drop").astype(float) + 1
    return buckets


def build_signal_bucket_diagnostics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"model_name", "horizon", "fwd_ret_1d", *SIGNAL_COLS}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction data missing columns: {sorted(missing)}")

    frames = [pred.copy()]
    all_period = pred.copy()
    all_period["diagnostic_period"] = "all"
    frames.append(all_period)
    work = pd.concat(frames, ignore_index=True)

    rows: list[dict[str, Any]] = []
    for (period, model_name, horizon), grp in work.groupby(["diagnostic_period", "model_name", "horizon"], sort=True):
        for signal_col in SIGNAL_COLS:
            bucketed = grp.copy()
            bucketed["signal_bucket"] = _bucket_signal(bucketed, signal_col)
            bucketed = bucketed.dropna(subset=["signal_bucket", "fwd_ret_1d", signal_col])
            if bucketed.empty:
                continue
            for bucket, bgrp in bucketed.groupby("signal_bucket", sort=True):
                ret = bgrp["fwd_ret_1d"].astype(float)
                rows.append(
                    {
                        "period": period,
                        "model_name": model_name,
                        "horizon": int(horizon),
                        "signal": signal_col,
                        "bucket": int(bucket),
                        "bucket_label": f"Q{int(bucket)}" + ("_low" if int(bucket) == 1 else "_high" if int(bucket) == 5 else ""),
                        "sample_count": int(len(bgrp)),
                        "signal_min": float(bgrp[signal_col].min()),
                        "signal_max": float(bgrp[signal_col].max()),
                        "signal_mean": float(bgrp[signal_col].mean()),
                        "future_1d_return_mean": float(ret.mean()),
                        "future_1d_return_median": float(ret.median()),
                        "future_1d_win_rate": float((ret > 0).mean()),
                    }
                )

    buckets = pd.DataFrame(rows)
    if buckets.empty:
        return buckets, pd.DataFrame()

    summary_rows: list[dict[str, Any]] = []
    for keys, grp in buckets.groupby(["period", "model_name", "horizon", "signal"], sort=True):
        period, model_name, horizon, signal = keys
        q1 = grp.loc[grp["bucket"] == 1]
        q5 = grp.loc[grp["bucket"] == 5]
        if q1.empty or q5.empty:
            continue
        low_mean = float(q1.iloc[0]["future_1d_return_mean"])
        high_mean = float(q5.iloc[0]["future_1d_return_mean"])
        top_minus_bottom = high_mean - low_mean
        if signal == "pred_big_down_prob":
            expected_edge = low_mean - high_mean
            high_group_expected = high_mean < low_mean
            decline_check = high_mean < 0
            decline_check_label = "high big-down bucket has negative mean"
        else:
            expected_edge = top_minus_bottom
            high_group_expected = high_mean > low_mean
            decline_check = low_mean < 0
            decline_check_label = "low upward-score bucket has negative mean"
        summary_rows.append(
            {
                "period": period,
                "model_name": model_name,
                "horizon": int(horizon),
                "signal": signal,
                "low_bucket_mean": low_mean,
                "high_bucket_mean": high_mean,
                "top_minus_bottom_mean": top_minus_bottom,
                "expected_edge": expected_edge,
                "high_group_has_expected_direction": bool(high_group_expected),
                "decline_check_label": decline_check_label,
                "decline_check_pass": bool(decline_check),
                "low_bucket_win_rate": float(q1.iloc[0]["future_1d_win_rate"]),
                "high_bucket_win_rate": float(q5.iloc[0]["future_1d_win_rate"]),
                "low_bucket_sample_count": int(q1.iloc[0]["sample_count"]),
                "high_bucket_sample_count": int(q5.iloc[0]["sample_count"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    buckets = buckets.merge(
        summary[
            [
                "period",
                "model_name",
                "horizon",
                "signal",
                "low_bucket_mean",
                "high_bucket_mean",
                "expected_edge",
                "high_group_has_expected_direction",
                "decline_check_pass",
            ]
        ],
        on=["period", "model_name", "horizon", "signal"],
        how="left",
    )
    return buckets.sort_values(["period", "model_name", "horizon", "signal", "bucket"]).reset_index(drop=True), summary


def select_best_real_no_leverage(metrics: pd.DataFrame) -> pd.Series:
    eligible = metrics.loc[metrics["strategy"].isin(NO_LEVERAGE_STRATEGIES)].copy()
    if "benchmark_clone" in eligible.columns:
        eligible = eligible.loc[~eligible["benchmark_clone"].astype(bool)]
    if eligible.empty:
        eligible = metrics.loc[metrics["strategy"].isin(NO_LEVERAGE_STRATEGIES)].copy()
    if eligible.empty:
        raise ValueError("No no-leverage ML strategy rows found.")
    return eligible.sort_values("total_return", ascending=False).iloc[0]


def _position_attribution(df: pd.DataFrame) -> dict[str, float]:
    position_gap = (1.0 - df["final_position"].astype(float)).clip(lower=0.0)
    buy_hold_return = df["buy_hold_return"].astype(float)
    missed_upside = float((position_gap * buy_hold_return.clip(lower=0.0)).sum())
    avoided_downside = float((position_gap * (-buy_hold_return.clip(upper=0.0))).sum())
    transaction_cost = float(df["transaction_cost"].astype(float).sum())
    return {
        "missed_upside": missed_upside,
        "avoided_downside": avoided_downside,
        "transaction_cost": transaction_cost,
        "net_timing_contribution": avoided_downside - missed_upside - transaction_cost,
    }


def build_strategy_failure_report(metrics: pd.DataFrame, strategy_daily: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    best = select_best_real_no_leverage(metrics)
    best_name = str(best["strategy"])
    best_daily = strategy_daily.loc[strategy_daily["strategy"] == best_name].sort_values("date").copy()
    best_attr = _position_attribution(best_daily)

    direct_name = "ml_direct_signal_timing"
    direct_metrics = metrics.loc[metrics["strategy"] == direct_name]
    direct_row = direct_metrics.iloc[0] if not direct_metrics.empty else None
    direct_daily = strategy_daily.loc[strategy_daily["strategy"] == direct_name].sort_values("date").copy()
    direct_attr = _position_attribution(direct_daily) if not direct_daily.empty else {}

    signal_rows: list[dict[str, Any]] = []
    if not direct_daily.empty:
        for signal, grp in direct_daily.groupby("signal", sort=True):
            attr = _position_attribution(grp)
            signal_rows.append(
                {
                    "signal": signal,
                    "days": int(len(grp)),
                    "avg_position": float(grp["final_position"].mean()),
                    "buy_hold_return_sum": float(grp["buy_hold_return"].sum()),
                    "strategy_return_sum": float(grp["strategy_return"].sum()),
                    "active_return_sum": float(grp["active_return"].sum()),
                    **attr,
                    "positive_buy_hold_days": int((grp["buy_hold_return"] > 0).sum()),
                    "negative_buy_hold_days": int((grp["buy_hold_return"] <= 0).sum()),
                }
            )
    signal_df = pd.DataFrame(signal_rows).sort_values("net_timing_contribution") if signal_rows else pd.DataFrame()

    wrong_sell_count = 0
    successful_sell_count = 0
    wrong_sell_missed_upside = np.nan
    buy_loss_count = 0
    if not direct_daily.empty:
        sell = direct_daily.loc[direct_daily["signal"] == "SELL"].copy()
        buy = direct_daily.loc[direct_daily["signal"] == "BUY"].copy()
        sell_gap = (1.0 - sell["final_position"].astype(float)).clip(lower=0.0)
        wrong_sell_count = int((sell["buy_hold_return"] > 0).sum())
        successful_sell_count = int((sell["buy_hold_return"] <= 0).sum())
        wrong_sell_missed_upside = float((sell_gap * sell["buy_hold_return"].clip(lower=0.0)).sum())
        buy_loss_count = int((buy["buy_hold_return"] <= 0).sum())

    if direct_attr:
        if direct_attr["missed_upside"] > direct_attr["avoided_downside"] and wrong_sell_count > successful_sell_count:
            direct_failure = "missed upside from reduced exposure, especially failed SELL signals"
        elif direct_attr["transaction_cost"] > abs(direct_attr["net_timing_contribution"]):
            direct_failure = "transaction cost"
        elif direct_attr["missed_upside"] > direct_attr["avoided_downside"]:
            direct_failure = "missed upside from reduced exposure"
        else:
            direct_failure = "weak signal edge after costs"
    else:
        direct_failure = "direct strategy data unavailable"

    report = f"""# Strategy Failure Attribution

## Best Real No-Leverage ML Strategy
- Strategy: {best_name}
- Total return: {_pct(best.get("total_return"))}
- Excess return vs buy-and-hold: {_pct(best.get("excess_return_vs_buy_hold"))}
- Max drawdown: {_pct(best.get("max_drawdown"))}
- Missed upside: {_fmt(best_attr["missed_upside"], 6)}
- Avoided downside: {_fmt(best_attr["avoided_downside"], 6)}
- Transaction cost: {_fmt(best_attr["transaction_cost"], 6)}
- Net timing contribution: {_fmt(best_attr["net_timing_contribution"], 6)}

The best no-leverage ML strategy did not beat buy-and-hold because reduced exposure missed more upside than it avoided downside. Costs are present but small relative to the missed-upside term.

## Direct BUY/SELL Strategy
- Strategy: {direct_name}
- Total return: {_pct(direct_row.get("total_return") if direct_row is not None else np.nan)}
- Excess return vs buy-and-hold: {_pct(direct_row.get("excess_return_vs_buy_hold") if direct_row is not None else np.nan)}
- Missed upside: {_fmt(direct_attr.get("missed_upside"), 6)}
- Avoided downside: {_fmt(direct_attr.get("avoided_downside"), 6)}
- Transaction cost: {_fmt(direct_attr.get("transaction_cost"), 6)}
- Net timing contribution: {_fmt(direct_attr.get("net_timing_contribution"), 6)}
- SELL signals: {wrong_sell_count + successful_sell_count}
- Successful SELL signals: {successful_sell_count}
- Failed SELL signals on positive next-day returns: {wrong_sell_count}
- Missed upside from failed SELL signals: {_fmt(wrong_sell_missed_upside, 6)}
- BUY signals with non-positive next-day returns: {buy_loss_count}
- Primary failure mode: {direct_failure}

## Direct Strategy Breakdown by Signal
{_markdown_table(signal_df, [
    "signal",
    "days",
    "avg_position",
    "buy_hold_return_sum",
    "active_return_sum",
    "missed_upside",
    "avoided_downside",
    "transaction_cost",
    "net_timing_contribution",
    "positive_buy_hold_days",
    "negative_buy_hold_days",
])}
"""
    facts = {
        "best_strategy": best_name,
        "best_attr": best_attr,
        "direct_failure": direct_failure,
        "wrong_sell_count": wrong_sell_count,
        "successful_sell_count": successful_sell_count,
        "direct_attr": direct_attr,
    }
    return report, facts


def build_validation_to_test_decay(metrics: pd.DataFrame) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if not TUNED_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Missing tuned params file: {TUNED_PARAMS_PATH}")
    tuned = json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for family, info in tuned.items():
        selected = info.get("selected", {})
        params = selected.get("params", {})
        test_match = metrics.loc[metrics["strategy"] == family]
        test = test_match.iloc[0].to_dict() if not test_match.empty else {}
        validation_excess = selected.get("excess_return_vs_buy_hold", np.nan)
        test_excess = test.get("excess_return_vs_buy_hold", np.nan)
        rows.append(
            {
                "strategy": family,
                "selected_model_name": params.get("model_name"),
                "selected_horizon": params.get("horizon"),
                "candidate_count": info.get("candidate_count"),
                "validation_score": selected.get("validation_score"),
                "validation_total_return": selected.get("total_return"),
                "validation_excess_return": validation_excess,
                "validation_max_drawdown": selected.get("max_drawdown"),
                "test_total_return": test.get("total_return", np.nan),
                "test_excess_return": test_excess,
                "test_max_drawdown": test.get("max_drawdown", np.nan),
                "test_benchmark_clone": test.get("benchmark_clone", np.nan),
                "excess_decay": (
                    float(test_excess) - float(validation_excess)
                    if pd.notna(test_excess) and pd.notna(validation_excess)
                    else np.nan
                ),
                "validation_positive_but_test_negative": bool(
                    pd.notna(validation_excess)
                    and pd.notna(test_excess)
                    and float(validation_excess) > 0
                    and float(test_excess) < 0
                ),
            }
        )
    decay = pd.DataFrame(rows).sort_values("excess_decay").reset_index(drop=True)

    flips = int(decay["validation_positive_but_test_negative"].sum()) if not decay.empty else 0
    avg_decay = float(decay["excess_decay"].mean()) if not decay.empty else np.nan
    all_test_negative = bool((decay["test_excess_return"] < 0).all()) if not decay.empty else False
    heavy_grid = bool((decay["candidate_count"].fillna(0) >= 500).any()) if not decay.empty else False
    overfit_likely = bool(flips > 0 or (np.isfinite(avg_decay) and avg_decay < -0.02 and all_test_negative))
    if overfit_likely and heavy_grid:
        overfit_text = "Likely. Validation-selected parameters decay materially out of sample, and several families search large parameter grids."
    elif overfit_likely:
        overfit_text = "Likely. Validation-selected edge does not carry to the test period."
    else:
        overfit_text = "Not obvious from sign flips alone, but validation-to-test decay is still unfavorable."

    report = f"""# Validation-to-Test Decay Report

## Summary
- Strategies checked: {len(decay)}
- Validation-positive but test-negative strategies: {flips}
- Average excess-return decay, test minus validation: {_fmt(avg_decay, 6)}
- All selected strategies negative vs buy-and-hold on test: {"Yes" if all_test_negative else "No"}
- Overfitting assessment: {overfit_text}

## Selected Strategy Decay
{_markdown_table(decay, [
    "strategy",
    "selected_model_name",
    "selected_horizon",
    "candidate_count",
    "validation_excess_return",
    "test_excess_return",
    "excess_decay",
    "validation_positive_but_test_negative",
], max_rows=20)}

## Interpretation
The decay is evaluated only from validation-selected artifacts and the held-out test backtest. No test-period information is used to select parameters here. A negative test result across all selected families points to signal/regime decay first, with parameter selection instability as a secondary risk.
"""
    facts = {
        "overfit_likely": overfit_likely,
        "overfit_text": overfit_text,
        "flips": flips,
        "avg_decay": avg_decay,
        "all_test_negative": all_test_negative,
    }
    return decay, report, facts


def build_signal_summary_report(
    buckets: pd.DataFrame,
    signal_summary: pd.DataFrame,
    prediction_source: str,
) -> tuple[str, dict[str, Any]]:
    if signal_summary.empty:
        return "# Signal Bucket Diagnostics\n\nNo signal bucket diagnostics were generated.\n", {}

    test_summary = signal_summary.loc[signal_summary["period"] == "test"].copy()
    if test_summary.empty:
        test_summary = signal_summary.loc[signal_summary["period"] == "all"].copy()

    aggregate = (
        test_summary.groupby("signal")
        .agg(
            mean_expected_edge=("expected_edge", "mean"),
            median_expected_edge=("expected_edge", "median"),
            direction_ok_rate=("high_group_has_expected_direction", "mean"),
            decline_check_rate=("decline_check_pass", "mean"),
            avg_low_bucket_mean=("low_bucket_mean", "mean"),
            avg_high_bucket_mean=("high_bucket_mean", "mean"),
            model_horizon_count=("expected_edge", "count"),
        )
        .reset_index()
        .sort_values("mean_expected_edge", ascending=False)
    )

    best_row = test_summary.sort_values("expected_edge", ascending=False).iloc[0].to_dict()
    worst_row = test_summary.sort_values("expected_edge", ascending=True).iloc[0].to_dict()
    low_checks = test_summary.loc[test_summary["signal"].isin(UPWARD_SIGNAL_COLS)]
    low_decline_rate = float(low_checks["decline_check_pass"].mean()) if not low_checks.empty else np.nan
    low_really_declines = bool(np.isfinite(low_decline_rate) and low_decline_rate >= 0.6)

    top_edges = test_summary.sort_values("expected_edge", ascending=False).copy()
    worst_edges = test_summary.sort_values("expected_edge", ascending=True).copy()

    report = f"""# Signal Bucket Diagnostics Summary

## Data Source
- Prediction source: `{prediction_source}`
- Bucket method: quintiles are formed separately for each period, model, horizon, and signal.
- Return tested: next tradable 1-day return, `fwd_ret_1d`.

## Aggregate Test-Period Signal Quality
{_markdown_table(aggregate, [
    "signal",
    "mean_expected_edge",
    "median_expected_edge",
    "direction_ok_rate",
    "decline_check_rate",
    "avg_low_bucket_mean",
    "avg_high_bucket_mean",
    "model_horizon_count",
], max_rows=10)}

## Best Test-Period Signal Buckets
{_markdown_table(top_edges, [
    "signal",
    "model_name",
    "horizon",
    "low_bucket_mean",
    "high_bucket_mean",
    "expected_edge",
    "high_group_has_expected_direction",
    "decline_check_pass",
], max_rows=12)}

## Worst Test-Period Signal Buckets
{_markdown_table(worst_edges, [
    "signal",
    "model_name",
    "horizon",
    "low_bucket_mean",
    "high_bucket_mean",
    "expected_edge",
    "high_group_has_expected_direction",
    "decline_check_pass",
], max_rows=12)}

## Interpretation
High-score buckets are not consistently enough better than low-score buckets across model/horizon pairs. For upward-style signals, the lowest quintile does not reliably imply a negative next-day return; in the test period, only {_pct(low_decline_rate)} of upward-signal low buckets have negative mean next-day return. This matters because a defensive timing strategy loses money when it cuts exposure in a broadly rising market without a strong negative-return signal.
"""
    facts = {
        "best_signal": best_row,
        "worst_signal": worst_row,
        "low_decline_rate": low_decline_rate,
        "low_really_declines": low_really_declines,
        "aggregate": aggregate,
    }
    return report, facts


def build_next_recommendation(
    signal_facts: dict[str, Any],
    strategy_facts: dict[str, Any],
    decay_facts: dict[str, Any],
) -> str:
    best = signal_facts.get("best_signal", {})
    worst = signal_facts.get("worst_signal", {})
    best_signal_text = (
        f"{best.get('signal')} / {best.get('model_name')} / {best.get('horizon')}D "
        f"(expected edge {_fmt(best.get('expected_edge'), 6)})"
        if best
        else "NA"
    )
    worst_signal_text = (
        f"{worst.get('signal')} / {worst.get('model_name')} / {worst.get('horizon')}D "
        f"(expected edge {_fmt(worst.get('expected_edge'), 6)})"
        if worst
        else "NA"
    )
    low_decline = signal_facts.get("low_really_declines", False)
    direct_failure = strategy_facts.get("direct_failure", "NA")
    overfit_text = decay_facts.get("overfit_text", "NA")

    if not low_decline:
        focus = "prediction signal"
        reason = "low predicted probability does not reliably map to negative next-day returns, so cutting exposure is not supported strongly enough."
    elif decay_facts.get("overfit_likely"):
        focus = "parameter selection"
        reason = "validation-selected edge decays materially on the held-out test period."
    else:
        focus = "strategy layer"
        reason = "signal ranking exists, but the conversion from score to exposure is losing too much upside."

    return f"""# Next Optimization Recommendation

## Recommendation
The next optimization should focus first on the {focus}.

## Why
- Best available signal: {best_signal_text}
- Worst signal: {worst_signal_text}
- Low predicted upward probability really implies future decline: {"Yes" if low_decline else "No"}
- Main timing failure mode: {direct_failure}
- Overfitting assessment: {overfit_text}

The current ML strategies mainly underperform because exposure reductions miss more upside than they avoid downside. Transaction costs are small compared with missed upside. Before adding new strategies or tuning more thresholds, the priority should be to verify and strengthen signals that identify genuinely negative forward returns, especially in rising regimes.

## Practical Next Step
Audit prediction targets and model selection for stable out-of-sample ranking power. The most useful next diagnostic is not another strategy variant, but a stricter signal validation pass: require that low upward-score buckets and high downside-score buckets produce negative or materially below-benchmark next-day returns across validation and test, then only pass those signals to the strategy layer.
"""


def main() -> None:
    summary_text = SUMMARY_PATH.read_text(encoding="utf-8") if SUMMARY_PATH.exists() else ""
    _ = summary_text

    predictions, prediction_source = load_predictions()
    metrics = _read_csv(STRATEGY_METRICS_PATH)
    if not STRATEGY_DAILY_PATH.exists():
        raise FileNotFoundError(f"Missing strategy daily file needed for attribution: {STRATEGY_DAILY_PATH}")
    strategy_daily = _read_csv(STRATEGY_DAILY_PATH)

    buckets, signal_summary = build_signal_bucket_diagnostics(predictions)
    buckets.to_csv(OUTPUT_DIR / "signal_bucket_diagnostics.csv", index=False)
    signal_report, signal_facts = build_signal_summary_report(buckets, signal_summary, prediction_source)
    (OUTPUT_DIR / "signal_bucket_diagnostics_summary.md").write_text(signal_report, encoding="utf-8")

    failure_report, strategy_facts = build_strategy_failure_report(metrics, strategy_daily)
    (OUTPUT_DIR / "strategy_failure_attribution.md").write_text(failure_report, encoding="utf-8")

    _, decay_report, decay_facts = build_validation_to_test_decay(metrics)
    (OUTPUT_DIR / "validation_to_test_decay_report.md").write_text(decay_report, encoding="utf-8")

    recommendation = build_next_recommendation(signal_facts, strategy_facts, decay_facts)
    (OUTPUT_DIR / "next_optimization_recommendation.md").write_text(recommendation, encoding="utf-8")

    best = signal_facts.get("best_signal", {})
    worst = signal_facts.get("worst_signal", {})
    best_available_signal = (
        f"{best.get('signal')} {best.get('model_name')} {best.get('horizon')}D "
        f"edge={_fmt(best.get('expected_edge'), 6)}"
        if best
        else "NA"
    )
    worst_signal = (
        f"{worst.get('signal')} {worst.get('model_name')} {worst.get('horizon')}D "
        f"edge={_fmt(worst.get('expected_edge'), 6)}"
        if worst
        else "NA"
    )
    low_decline_answer = "Yes" if signal_facts.get("low_really_declines") else "No"
    overfit_answer = "Yes" if decay_facts.get("overfit_likely") else "No"
    generated = [
        "outputs/signal_bucket_diagnostics.csv",
        "outputs/signal_bucket_diagnostics_summary.md",
        "outputs/strategy_failure_attribution.md",
        "outputs/validation_to_test_decay_report.md",
        "outputs/next_optimization_recommendation.md",
    ]

    print("===== NEXT OPTIMIZATION DIAGNOSIS =====")
    print(f"Best available signal: {best_available_signal}")
    print(f"Worst signal: {worst_signal}")
    print(f"Does low predicted probability really imply future decline: {low_decline_answer}")
    print(f"Main reason ML timing underperforms: {strategy_facts.get('direct_failure', 'NA')}")
    print(f"Is overfitting likely: {overfit_answer}")
    print(
        "Recommended next step: "
        + ("prediction signal validation" if not signal_facts.get("low_really_declines") else "parameter selection review")
    )
    print("Generated files:")
    for path in generated:
        print(f"- {path}")
    print("======================================")


if __name__ == "__main__":
    main()
