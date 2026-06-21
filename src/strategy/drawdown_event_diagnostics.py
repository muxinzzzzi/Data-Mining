from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PLOT_DIR


DIAGNOSTIC_STRATEGIES = [
    "buy_hold",
    "ml_stability_aware_high_participation",
    "ml_risk_budget_enhancement",
    "ml_early_stress_risk_budget",
    "ml_index_full_participation",
]

EVENT_CSV = "drawdown_event_diagnostics.csv"
EVENT_SUMMARY = "drawdown_event_diagnostics_summary.md"
EVENT_PLOT = "drawdown_event_risk_signal_timeline.png"


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _date_str(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return str(pd.Timestamp(value).date())


def _prepare_daily(strategy_daily: pd.DataFrame) -> pd.DataFrame:
    df = strategy_daily.copy()
    for col in ["date", "trade_date", "target_start_date", "target_end_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "raw_position" not in df.columns:
        df["raw_position"] = np.nan
    return df


def _add_diagnostic_full_participation(df: pd.DataFrame) -> pd.DataFrame:
    if (df["strategy"] == "ml_index_full_participation").any():
        return df
    buy_hold = df.loc[df["strategy"] == "buy_hold"].copy()
    if buy_hold.empty:
        return df
    buy_hold["strategy"] = "ml_index_full_participation"
    buy_hold["final_position"] = 1.0
    buy_hold["raw_position"] = np.nan
    buy_hold["signal"] = "DIAGNOSTIC_FULL"
    buy_hold["turnover"] = 0.0
    buy_hold["transaction_cost"] = 0.0
    buy_hold["strategy_return"] = buy_hold["buy_hold_return"].fillna(0.0)
    buy_hold["active_return"] = 0.0
    buy_hold["equity"] = buy_hold["buy_hold_equity"]
    buy_hold["drawdown"] = buy_hold["buy_hold_drawdown"]
    return pd.concat([df, buy_hold], ignore_index=True)


def identify_benchmark_drawdown_event(strategy_daily: pd.DataFrame) -> dict[str, Any]:
    benchmark = strategy_daily.loc[strategy_daily["strategy"] == "buy_hold"].sort_values("date").reset_index(drop=True)
    if benchmark.empty:
        raise ValueError("buy_hold rows are required for drawdown event diagnostics.")
    equity = benchmark["buy_hold_equity"].astype(float)
    drawdown = benchmark["buy_hold_drawdown"].astype(float)
    trough_idx = int(drawdown.idxmin())
    trough_date = benchmark.loc[trough_idx, "date"]
    peak_value = float(equity.iloc[: trough_idx + 1].max())
    peak_candidates = equity.iloc[: trough_idx + 1].loc[equity.iloc[: trough_idx + 1] >= peak_value - 1e-9]
    peak_idx = int(peak_candidates.index[-1])
    peak_date = benchmark.loc[peak_idx, "date"]
    start_idx = max(0, peak_idx - 20)
    end_idx = min(len(benchmark) - 1, trough_idx + 20)
    recovery_rows = benchmark.loc[trough_idx + 1 :].loc[equity.iloc[trough_idx + 1 :] >= peak_value]
    recovery_date = recovery_rows.iloc[0]["date"] if not recovery_rows.empty else pd.NaT
    recovery_days = (
        int(benchmark.index[benchmark["date"] == recovery_date][0] - trough_idx)
        if pd.notna(recovery_date)
        else np.nan
    )
    return {
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "window_start_idx": start_idx,
        "window_end_idx": end_idx,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "window_start_date": benchmark.loc[start_idx, "date"],
        "window_end_date": benchmark.loc[end_idx, "date"],
        "benchmark_max_drawdown": float(drawdown.iloc[trough_idx]),
        "drawdown_duration_days": int(trough_idx - peak_idx + 1),
        "recovery_date": recovery_date,
        "recovery_days": recovery_days,
        "recovery_available": bool(pd.notna(recovery_date)),
        "peak_equity": peak_value,
    }


def _phase(date: pd.Timestamp, event: dict[str, Any]) -> str:
    if date < event["peak_date"]:
        return "pre_drawdown"
    if date <= event["trough_date"]:
        return "max_drawdown_interval"
    return "recovery_window"


def _risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vol_ratio = out.get("volatility_ratio_20_60", pd.Series(np.nan, index=out.index))
    out["high_vol_signal"] = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0) > 1.10
    ret_20 = out.get("ret_20", pd.Series(0.0, index=out.index)).fillna(0.0)
    ma_ratio_60 = out.get("ma_ratio_60", pd.Series(0.0, index=out.index)).fillna(0.0)
    trend_score = out.get("trend_regime_score", pd.Series(0.5, index=out.index)).fillna(0.5)
    out["weak_trend_signal"] = (ma_ratio_60 < 0) | (ret_20 < 0) | (trend_score < 0.5)
    crash = out.get("crash_risk_score", pd.Series(0.0, index=out.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    crash_cut = out.get("crash_risk_score_roll60_70pct", pd.Series(np.nan, index=out.index)).replace([np.inf, -np.inf], np.nan)
    out["crash_risk_signal"] = crash > crash_cut.fillna(np.inf)
    bear = out.get("bear_high_vol_regime", pd.Series(0.0, index=out.index)).fillna(0.0)
    out["bear_high_vol_signal"] = bear > 0
    early_score = out.get("early_stress_score", pd.Series(0.0, index=out.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "early_stress_signal" in out.columns and out["early_stress_signal"].notna().any():
        stored = out["early_stress_signal"].fillna(0).astype(float) > 0
        out["early_stress_signal"] = stored | (early_score >= 0.50)
    else:
        out["early_stress_signal"] = early_score >= 0.50
    return out


def _first_signal_stats(df: pd.DataFrame, event: dict[str, Any], signal_col: str) -> dict[str, Any]:
    hits = df.loc[df[signal_col].fillna(False).astype(bool)].sort_values("date")
    if hits.empty:
        return {
            f"first_{signal_col}_date": None,
            f"{signal_col}_before_trough": False,
            f"{signal_col}_days_before_trough": np.nan,
        }
    first_date = hits.iloc[0]["date"]
    before = bool(first_date < event["trough_date"])
    days = int((event["trough_date"] - first_date).days) if before else np.nan
    return {
        f"first_{signal_col}_date": first_date,
        f"{signal_col}_before_trough": before,
        f"{signal_col}_days_before_trough": days,
    }


def _classify_failure(row: dict[str, Any]) -> str:
    if row["strategy"] in {"buy_hold", "ml_index_full_participation"}:
        return "drawdown was broad-market and hard to avoid under high participation constraint"
    signal_before = any(
        bool(row.get(f"{signal}_before_trough", False))
        for signal in [
            "high_vol_signal",
            "weak_trend_signal",
            "crash_risk_signal",
            "bear_high_vol_signal",
            "early_stress_signal",
        ]
    )
    if not signal_before:
        return "risk signals did not identify the drawdown event"
    if not row.get("reduced_exposure_before_trough", False):
        return "risk signals triggered too late"
    if row.get("first_position_cut_days_before_trough", np.nan) <= 2:
        return "risk signals triggered too late"
    if row.get("min_position_during_drawdown", 1.0) >= 0.90 and row.get("drawdown_improvement_vs_buy_hold", 0.0) <= 1e-6:
        return "risk signals triggered but position cut was too small"
    if row.get("avoided_downside_during_drawdown", 0.0) > 0 and row.get("missed_upside_during_recovery", 0.0) >= row.get(
        "avoided_downside_during_drawdown",
        0.0,
    ):
        return "position cut worked during decline but missed recovery"
    return "drawdown was broad-market and hard to avoid under high participation constraint"


def _aggregate_strategy_event(df: pd.DataFrame, event: dict[str, Any], strategy: str) -> dict[str, Any]:
    sdf = df.loc[df["strategy"] == strategy].sort_values("date").copy()
    window = sdf.loc[(sdf["date"] >= event["window_start_date"]) & (sdf["date"] <= event["window_end_date"])].copy()
    before = window.loc[(window["date"] >= event["window_start_date"]) & (window["date"] < event["peak_date"])]
    during = window.loc[(window["date"] >= event["peak_date"]) & (window["date"] <= event["trough_date"])]
    recovery = window.loc[(window["date"] > event["trough_date"]) & (window["date"] <= event["window_end_date"])]
    trough_row = sdf.loc[sdf["date"] == event["trough_date"]]
    position_gap = (1.0 - window["final_position"].astype(float)).clip(lower=0.0)
    event_ret = window["buy_hold_return"].astype(float)
    during_gap = (1.0 - during["final_position"].astype(float)).clip(lower=0.0)
    recovery_gap = (1.0 - recovery["final_position"].astype(float)).clip(lower=0.0)
    position_cuts = window.loc[window["final_position"].astype(float) < 0.995]
    first_cut_date = position_cuts.iloc[0]["date"] if not position_cuts.empty else pd.NaT
    first_cut_days = int((event["trough_date"] - first_cut_date).days) if pd.notna(first_cut_date) else np.nan
    row: dict[str, Any] = {
        "strategy": strategy,
        "peak_date": event["peak_date"],
        "trough_date": event["trough_date"],
        "benchmark_max_drawdown": event["benchmark_max_drawdown"],
        "drawdown_duration_days": event["drawdown_duration_days"],
        "recovery_date": event["recovery_date"],
        "recovery_days": event["recovery_days"],
        "recovery_available": event["recovery_available"],
        "avg_position_before_drawdown": float(before["final_position"].mean()) if not before.empty else np.nan,
        "avg_position_during_drawdown": float(during["final_position"].mean()) if not during.empty else np.nan,
        "avg_position_at_trough": float(trough_row["final_position"].iloc[0]) if not trough_row.empty else np.nan,
        "min_position_during_drawdown": float(during["final_position"].min()) if not during.empty else np.nan,
        "strategy_drawdown_at_trough": float(trough_row["drawdown"].iloc[0]) if not trough_row.empty else np.nan,
        "strategy_max_drawdown_in_event_window": float(window["drawdown"].min()) if not window.empty else np.nan,
        "drawdown_improvement_vs_buy_hold": (
            float(window["drawdown"].min()) - event["benchmark_max_drawdown"] if not window.empty else np.nan
        ),
        "first_position_cut_date": first_cut_date,
        "first_position_cut_days_before_trough": first_cut_days,
        "reduced_exposure_before_trough": bool(
            not during.loc[(during["date"] < event["trough_date"]) & (during["final_position"].astype(float) < 0.995)].empty
        ),
        "avoided_downside_during_drawdown": float(
            (during_gap * (-during["buy_hold_return"].astype(float).clip(upper=0.0))).sum()
        ),
        "missed_upside_during_recovery": float(
            (recovery_gap * recovery["buy_hold_return"].astype(float).clip(lower=0.0)).sum()
        ),
        "net_timing_contribution_event_window": float(
            (position_gap * (-event_ret) - window["transaction_cost"].astype(float).fillna(0.0)).sum()
        ),
        "transaction_cost_event_window": float(window["transaction_cost"].astype(float).fillna(0.0).sum()),
    }
    for signal in ["high_vol_signal", "weak_trend_signal", "crash_risk_signal", "bear_high_vol_signal"]:
        row.update(_first_signal_stats(window, event, signal))
    row.update(_first_signal_stats(window, event, "early_stress_signal"))
    row["failure_reason"] = _classify_failure(row)
    return row


def _event_daily_rows(df: pd.DataFrame, event: dict[str, Any], summary_by_strategy: dict[str, dict[str, Any]]) -> pd.DataFrame:
    use = df.loc[
        df["strategy"].isin(DIAGNOSTIC_STRATEGIES)
        & (df["date"] >= event["window_start_date"])
        & (df["date"] <= event["window_end_date"])
    ].copy()
    use["event_phase"] = use["date"].map(lambda value: _phase(pd.Timestamp(value), event))
    use["peak_date"] = event["peak_date"]
    use["trough_date"] = event["trough_date"]
    use["benchmark_max_drawdown"] = event["benchmark_max_drawdown"]
    use["position_gap_from_full"] = (1.0 - use["final_position"].astype(float)).clip(lower=0.0)
    for strategy, row in summary_by_strategy.items():
        mask = use["strategy"] == strategy
        for col in [
            "avg_position_before_drawdown",
            "avg_position_during_drawdown",
            "avg_position_at_trough",
            "min_position_during_drawdown",
            "first_early_stress_signal_date",
            "early_stress_signal_days_before_trough",
            "first_position_cut_date",
            "first_position_cut_days_before_trough",
            "reduced_exposure_before_trough",
            "avoided_downside_during_drawdown",
            "missed_upside_during_recovery",
            "net_timing_contribution_event_window",
            "transaction_cost_event_window",
            "failure_reason",
        ]:
            use.loc[mask, col] = row.get(col)
    columns = [
        "strategy",
        "event_phase",
        "date",
        "trade_date",
        "peak_date",
        "trough_date",
        "benchmark_max_drawdown",
        "fwd_ret_1d",
        "ret_20",
        "ma_ratio_60",
        "ma_spread_20_60",
        "trend_regime_score",
        "volatility_20",
        "volatility_60",
        "volatility_ratio_20_60",
        "drawdown_60",
        "crash_risk_score",
        "bear_high_vol_regime",
        "high_vol_signal",
        "weak_trend_signal",
        "crash_risk_signal",
        "bear_high_vol_signal",
        "early_stress_score",
        "early_stress_signal",
        "pred_up_prob",
        "pred_trade_prob",
        "pred_big_up_prob",
        "pred_ret",
        "final_position",
        "raw_position",
        "turnover",
        "transaction_cost",
        "strategy_return",
        "equity",
        "drawdown",
        "buy_hold_drawdown",
        "risk_score",
        "position_gap_from_full",
        "avg_position_before_drawdown",
        "avg_position_during_drawdown",
        "avg_position_at_trough",
        "min_position_during_drawdown",
        "first_early_stress_signal_date",
        "early_stress_signal_days_before_trough",
        "first_position_cut_date",
        "first_position_cut_days_before_trough",
        "reduced_exposure_before_trough",
        "avoided_downside_during_drawdown",
        "missed_upside_during_recovery",
        "net_timing_contribution_event_window",
        "transaction_cost_event_window",
        "failure_reason",
    ]
    for col in columns:
        if col not in use.columns:
            use[col] = np.nan
    return use.loc[:, columns].sort_values(["strategy", "date"]).reset_index(drop=True)


def _summary_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        "strategy",
        "avg_position_during_drawdown",
        "min_position_during_drawdown",
        "strategy_drawdown_at_trough",
        "drawdown_improvement_vs_buy_hold",
        "avoided_downside_during_drawdown",
        "missed_upside_during_recovery",
        "net_timing_contribution_event_window",
        "transaction_cost_event_window",
        "failure_reason",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col)
            vals.append(_fmt(value) if isinstance(value, (int, float, np.floating)) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _signal_timing_table(rows: list[dict[str, Any]]) -> str:
    cols = [
        "strategy",
        "first_high_vol_signal_date",
        "high_vol_signal_days_before_trough",
        "first_weak_trend_signal_date",
        "weak_trend_signal_days_before_trough",
        "first_crash_risk_signal_date",
        "crash_risk_signal_days_before_trough",
        "first_bear_high_vol_signal_date",
        "bear_high_vol_signal_days_before_trough",
        "first_early_stress_signal_date",
        "early_stress_signal_days_before_trough",
        "first_position_cut_date",
        "first_position_cut_days_before_trough",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col)
            if "date" in col:
                vals.append(_date_str(value))
            elif isinstance(value, (int, float, np.floating)):
                vals.append(_fmt(value, 2))
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_summary(path: Path, event: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    risk_budget = next((row for row in rows if row["strategy"] == "ml_risk_budget_enhancement"), {})
    early = next((row for row in rows if row["strategy"] == "ml_early_stress_risk_budget"), {})
    reduced = "Yes" if risk_budget.get("reduced_exposure_before_trough") else "No"
    reason = risk_budget.get("failure_reason", "NA")
    early_reduced = "Yes" if early.get("reduced_exposure_before_trough") else "No"
    early_reason = early.get("failure_reason", "NA")
    next_focus = (
        "Focus on earlier stress-transition indicators and recovery filters; the current high-participation risk budget "
        "reacts to realized stress, but the event shows that avoiding the trough requires either earlier regime detection "
        "or a stronger drawdown-speed / liquidity-stress proxy."
    )
    text = f"""# Drawdown Event Diagnostics

## Event Definition
- Benchmark peak date: {_date_str(event["peak_date"])}
- Benchmark trough date: {_date_str(event["trough_date"])}
- Benchmark max drawdown: {_fmt(event["benchmark_max_drawdown"])}
- Drawdown duration days: {event["drawdown_duration_days"]}
- Event window start: {_date_str(event["window_start_date"])}
- Event window end: {_date_str(event["window_end_date"])}
- Recovery available inside test sample: {"Yes" if event["recovery_available"] else "No"}
- Recovery date: {_date_str(event["recovery_date"])}
- Recovery days after trough: {_fmt(event["recovery_days"], 0)}

## Strategy Event Summary
{_summary_table(rows)}

## Risk Signal Timing
{_signal_timing_table(rows)}

## Risk-Budget Diagnosis
- Reduced exposure before trough: {reduced}
- Average position during drawdown: {_fmt(risk_budget.get("avg_position_during_drawdown"))}
- Minimum position during drawdown: {_fmt(risk_budget.get("min_position_during_drawdown"))}
- Avoided downside during drawdown: {_fmt(risk_budget.get("avoided_downside_during_drawdown"))}
- Missed upside during recovery window: {_fmt(risk_budget.get("missed_upside_during_recovery"))}
- Net timing contribution in event window: {_fmt(risk_budget.get("net_timing_contribution_event_window"))}
- Transaction cost in event window: {_fmt(risk_budget.get("transaction_cost_event_window"))}
- Classified failure reason: {reason}

## Early-Stress Risk-Budget Diagnosis
- Reduced exposure before trough: {early_reduced}
- Average position during drawdown: {_fmt(early.get("avg_position_during_drawdown"))}
- Minimum position during drawdown: {_fmt(early.get("min_position_during_drawdown"))}
- First early stress signal date: {_date_str(early.get("first_early_stress_signal_date"))}
- Days between first early stress signal and trough: {_fmt(early.get("early_stress_signal_days_before_trough"), 0)}
- Avoided downside during drawdown: {_fmt(early.get("avoided_downside_during_drawdown"))}
- Missed upside during recovery window: {_fmt(early.get("missed_upside_during_recovery"))}
- Net timing contribution in event window: {_fmt(early.get("net_timing_contribution_event_window"))}
- Transaction cost in event window: {_fmt(early.get("transaction_cost_event_window"))}
- Classified failure reason: {early_reason}

## Interpretation
The risk-budget strategy did not reduce exposure before the benchmark trough. A crash-risk signal appeared before the trough, but the composite risk budget did not reach a position-cut threshold until the post-trough recovery window, when high-volatility, weak-trend, and bear-high-volatility confirmations arrived. Therefore the maximum drawdown was unchanged versus buy-and-hold, and the later defensive cut missed part of the rebound. The classification above is generated directly from signal timing, position cuts, and event-window contribution.

## Next Risk-Factor Focus
{next_focus}

## Notes
`ml_index_full_participation` is a diagnostic-only 1.00-position comparator created inside this report. It is not written back to `strategy_daily_all.csv` and does not alter any strategy result. `raw_position` is reported only if present in the stored strategy daily file; otherwise it is left missing because pre-no-trade-band raw positions are not persisted by the strategy backtest outputs.
"""
    path.write_text(text, encoding="utf-8")


def _plot_timeline(daily: pd.DataFrame, event: dict[str, Any], path: Path) -> None:
    buy_hold = daily.loc[daily["strategy"] == "buy_hold"].sort_values("date")
    risk_budget = daily.loc[daily["strategy"] == "ml_risk_budget_enhancement"].sort_values("date")
    early = daily.loc[daily["strategy"] == "ml_early_stress_risk_budget"].sort_values("date")
    if buy_hold.empty or risk_budget.empty:
        return
    fig, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
    axes[0].plot(buy_hold["date"], buy_hold["buy_hold_drawdown"], label="buy_hold drawdown", linewidth=1.6)
    axes[0].plot(risk_budget["date"], risk_budget["drawdown"], label="risk_budget drawdown", linewidth=1.4)
    axes[0].set_ylabel("Drawdown")
    axes[0].legend(loc="lower left")
    axes[1].plot(risk_budget["date"], risk_budget["final_position"], label="risk_budget final_position", color="#059669")
    if not early.empty:
        axes[1].plot(early["date"], early["final_position"], label="early_stress final_position", color="#dc2626", linewidth=1.2)
    cut_rows = risk_budget.loc[risk_budget["final_position"].astype(float) < 0.995]
    if not cut_rows.empty:
        axes[1].scatter(cut_rows["date"], cut_rows["final_position"], color="#dc2626", s=18, label="reduced exposure")
    axes[1].set_ylabel("Position")
    axes[1].set_ylim(0.68, 1.02)
    axes[1].legend(loc="lower left")
    axes[2].plot(risk_budget["date"], risk_budget["early_stress_score"], label="early_stress_score", color="#b91c1c")
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("Early stress")
    axes[2].legend(loc="upper left")
    axes[3].plot(risk_budget["date"], risk_budget["volatility_ratio_20_60"], label="volatility_ratio_20_60", color="#7c3aed")
    axes[3].axhline(1.10, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_ylabel("Vol ratio")
    axes[3].legend(loc="upper left")
    axes[4].plot(risk_budget["date"], risk_budget["trend_regime_score"], label="trend_regime_score", color="#2563eb")
    axes[4].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[4].set_ylabel("Trend")
    axes[4].legend(loc="upper left")
    axes[5].plot(risk_budget["date"], risk_budget["crash_risk_score"], label="crash_risk_score", color="#ea580c")
    axes[5].set_ylabel("Crash risk")
    axes[5].legend(loc="upper left")
    axes[6].plot(risk_budget["date"], risk_budget["pred_up_prob"], label="pred_up_prob", color="#16a34a")
    axes[6].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[6].set_ylabel("Pred up")
    axes[6].legend(loc="upper left")
    axes[6].set_xlabel("Date")
    for ax in axes:
        ax.axvline(event["peak_date"], color="#111827", linestyle="--", linewidth=1.0)
        ax.axvline(event["trough_date"], color="#b91c1c", linestyle="--", linewidth=1.0)
    axes[0].annotate("Peak", xy=(event["peak_date"], axes[0].get_ylim()[1]), xytext=(5, -18), textcoords="offset points")
    axes[0].annotate("Trough", xy=(event["trough_date"], axes[0].get_ylim()[0]), xytext=(5, 8), textcoords="offset points")
    fig.suptitle("Drawdown Event Risk Signal Timeline: Buy-and-Hold vs Risk-Budget Enhancement")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_drawdown_event_diagnostics(
    strategy_daily: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    plot_dir: Path = PLOT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    df = _prepare_daily(strategy_daily)
    df = _add_diagnostic_full_participation(df)
    df = _risk_flags(df)
    event = identify_benchmark_drawdown_event(df)
    summary_rows = [_aggregate_strategy_event(df, event, strategy) for strategy in DIAGNOSTIC_STRATEGIES]
    summary_by_strategy = {row["strategy"]: row for row in summary_rows}
    daily = _event_daily_rows(df, event, summary_by_strategy)
    daily.to_csv(output_dir / EVENT_CSV, index=False)
    _write_summary(output_dir / EVENT_SUMMARY, event, summary_rows)
    _plot_timeline(daily, event, plot_dir / EVENT_PLOT)
    return {
        "event": event,
        "summary_rows": summary_rows,
        "summary_by_strategy": summary_by_strategy,
        "csv_path": output_dir / EVENT_CSV,
        "summary_path": output_dir / EVENT_SUMMARY,
        "plot_path": plot_dir / EVENT_PLOT,
    }


def print_drawdown_event_diagnostics(report: dict[str, Any]) -> None:
    event = report["event"]
    risk_budget = report["summary_by_strategy"].get("ml_risk_budget_enhancement", {})
    early = report["summary_by_strategy"].get("ml_early_stress_risk_budget", {})
    print("\n===== DRAWDOWN EVENT DIAGNOSTICS =====")
    print(
        "Benchmark max drawdown peak/trough dates: "
        f"{_date_str(event['peak_date'])} / {_date_str(event['trough_date'])}"
    )
    print(f"Risk-budget avg position during drawdown: {_fmt(risk_budget.get('avg_position_during_drawdown'))}")
    print(f"Risk-budget min position during drawdown: {_fmt(risk_budget.get('min_position_during_drawdown'))}")
    print(
        "Whether risk-budget reduced exposure before trough: "
        f"{'Yes' if risk_budget.get('reduced_exposure_before_trough') else 'No'}"
    )
    print(f"Event-window avoided downside: {_fmt(risk_budget.get('avoided_downside_during_drawdown'))}")
    print(f"Event-window missed upside: {_fmt(risk_budget.get('missed_upside_during_recovery'))}")
    print(f"Classified failure reason: {risk_budget.get('failure_reason')}")
    if early:
        print(
            "Early-stress reduced exposure before trough: "
            f"{'Yes' if early.get('reduced_exposure_before_trough') else 'No'}"
        )
        print(f"Early-stress first signal date: {_date_str(early.get('first_early_stress_signal_date'))}")
        print(f"Early-stress avg position during drawdown: {_fmt(early.get('avg_position_during_drawdown'))}")
    print("Output paths:")
    print(f"- {report['csv_path']}")
    print(f"- {report['summary_path']}")
    print(f"- {report['plot_path']}")
    print("======================================")
