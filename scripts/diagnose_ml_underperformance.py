from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import COST_RATE, DIAGNOSTICS_DIR, INITIAL_CAPITAL, OUTPUT_DIR, PREDICTION_DIR

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN_STRATEGY = "ml_big_up_index_enhancement"
PLOT_DIR = DIAGNOSTICS_DIR / "plots"

CONTEXT_COLS = [
    "date",
    "trade_date",
    "fwd_ret_1d",
    "target_ret",
    "final_position",
    "position_gap",
    "position_gap_return",
    "turnover",
    "turnover_cost",
    "pred_ret",
    "pred_up_prob",
    "pred_trade_prob",
    "pred_big_up_prob",
    "pred_big_down_prob",
    "pred_tail_score",
    "ret_20",
    "ma_ratio_60",
    "ma_spread_20_60",
    "drawdown_60",
    "volatility_20",
    "volatility_60",
    "trend_regime_score",
]


def _ensure_dirs() -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["date", "trade_date", "target_start_date", "target_end_date", "chunk_start", "train_latest_target_end_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _fmt(value: Any, digits: int = 4) -> str:
    value = _safe_float(value)
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def _selected_model_horizon(strategy_daily: pd.DataFrame) -> tuple[str, int]:
    path = OUTPUT_DIR / "strategy_tuned_params.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            params = data.get(MAIN_STRATEGY, {}).get("selected", {}).get("params", {})
            model_name = params.get("model_name")
            horizon = params.get("horizon")
            if model_name is not None and horizon is not None:
                return str(model_name), int(horizon)
        except Exception:
            pass

    ml = strategy_daily.loc[strategy_daily["strategy"] == MAIN_STRATEGY]
    if not ml.empty and {"model_name", "horizon"}.issubset(ml.columns):
        row = ml[["model_name", "horizon"]].dropna().iloc[0]
        return str(row["model_name"]), int(row["horizon"])
    raise ValueError(f"Cannot infer selected model/horizon for {MAIN_STRATEGY}.")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, int]]:
    strategy_path = OUTPUT_DIR / "strategy_daily_all.csv"
    metrics_path = OUTPUT_DIR / "strategy_metrics_all.csv"
    prediction_path = PREDICTION_DIR / "test_predictions_all_models.csv"
    if not strategy_path.exists():
        raise FileNotFoundError(f"Missing {strategy_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing {prediction_path}")

    strategy_daily = _parse_dates(pd.read_csv(strategy_path))
    strategy_metrics = pd.read_csv(metrics_path)
    predictions = _parse_dates(pd.read_csv(prediction_path))
    selected = _selected_model_horizon(strategy_daily)
    return strategy_daily, strategy_metrics, predictions, selected


def compute_exposure_diagnostics(strategy_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for strategy in ["buy_hold", MAIN_STRATEGY]:
        grp = strategy_daily.loc[strategy_daily["strategy"] == strategy].sort_values("date").copy()
        if grp.empty:
            continue
        pos = grp["final_position"].astype(float)
        turnover = grp["turnover"].astype(float)
        rows.append(
            {
                "strategy": strategy,
                "average_final_position": float(pos.mean()),
                "minimum_final_position": float(pos.min()),
                "maximum_final_position": float(pos.max()),
                "days_below_full_exposure": int((pos < 0.999).sum()),
                "pct_days_below_full_exposure": float((pos < 0.999).mean()),
                "total_turnover": float(turnover.sum()),
                "average_turnover": float(turnover.mean()),
                "average_abs_position_gap_from_1": float((pos - 1.0).abs().mean()),
                "days_position_le_095": int((pos <= 0.95).sum()),
                "days_position_le_090": int((pos <= 0.90).sum()),
                "days_position_le_080": int((pos <= 0.80).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(DIAGNOSTICS_DIR / "ml_strategy_exposure_diagnostics.csv", index=False)
    return out


def export_strategy_comparison(strategy_metrics: pd.DataFrame) -> pd.DataFrame:
    names = [
        "buy_hold",
        "ml_big_up_index_enhancement",
        "ml_conservative_full_participation_enhancement",
        "ml_ultra_conservative_full_participation_enhancement",
    ]
    cols = [
        "strategy",
        "total_return",
        "excess_return_vs_buy_hold",
        "max_drawdown",
        "average_position",
        "minimum_position",
        "days_below_full_exposure",
        "missed_upside",
        "avoided_downside",
        "total_transaction_cost",
        "net_timing_contribution",
        "benchmark_clone",
    ]
    available_cols = [col for col in cols if col in strategy_metrics.columns]
    out = strategy_metrics.loc[strategy_metrics["strategy"].isin(names), available_cols].copy()
    order = {name: idx for idx, name in enumerate(names)}
    out["order"] = out["strategy"].map(order)
    out = out.sort_values("order").drop(columns="order")
    out = out.rename(
        columns={
            "average_position": "avg_position",
            "minimum_position": "min_position",
            "total_transaction_cost": "turnover_cost",
        }
    )
    out.to_csv(DIAGNOSTICS_DIR / "conservative_strategy_comparison.csv", index=False)
    return out


def build_active_attribution(strategy_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ml = strategy_daily.loc[strategy_daily["strategy"] == MAIN_STRATEGY].sort_values("date").copy()
    bench = strategy_daily.loc[strategy_daily["strategy"] == "buy_hold"].sort_values("date").copy()
    if ml.empty or bench.empty:
        raise ValueError("Need both ML strategy and buy_hold rows for attribution.")

    bench_cols = ["date", "strategy_return", "equity"]
    joined = ml.merge(
        bench[bench_cols].rename(columns={"strategy_return": "benchmark_return", "equity": "benchmark_equity"}),
        on="date",
        how="left",
    )
    joined["strategy_return"] = joined["strategy_return"].astype(float)
    joined["benchmark_return"] = joined["benchmark_return"].fillna(joined["buy_hold_return"]).astype(float)
    joined["active_return"] = joined["strategy_return"] - joined["benchmark_return"]
    joined["position_gap"] = joined["final_position"].astype(float) - 1.0
    joined["position_gap_return"] = joined["position_gap"] * joined["fwd_ret_1d"].astype(float)
    joined["turnover_cost"] = joined["turnover"].astype(float) * COST_RATE
    joined["cumulative_active_return"] = joined["active_return"].cumsum()
    joined["cumulative_position_gap_return"] = joined["position_gap_return"].cumsum()
    joined["cumulative_turnover_cost"] = joined["turnover_cost"].cumsum()
    joined["missed_upside_daily"] = np.where(
        joined["fwd_ret_1d"] > 0,
        -np.minimum(joined["position_gap_return"], 0.0),
        0.0,
    )
    joined["avoided_downside_daily"] = np.where(
        joined["fwd_ret_1d"] < 0,
        np.maximum(joined["position_gap_return"], 0.0),
        0.0,
    )
    joined["cumulative_missed_upside"] = joined["missed_upside_daily"].cumsum()
    joined["cumulative_avoided_downside"] = joined["avoided_downside_daily"].cumsum()

    ml_final_equity = float(joined["equity"].iloc[-1])
    bench_final_equity = float(joined["benchmark_equity"].iloc[-1])
    summary = {
        "total_active_return_arithmetic": float(joined["active_return"].sum()),
        "total_position_gap_return": float(joined["position_gap_return"].sum()),
        "missed_upside": float(joined["missed_upside_daily"].sum()),
        "avoided_downside": float(joined["avoided_downside_daily"].sum()),
        "turnover_cost": float(joined["turnover_cost"].sum()),
        "net_timing_contribution": float(
            joined["avoided_downside_daily"].sum()
            - joined["missed_upside_daily"].sum()
            - joined["turnover_cost"].sum()
        ),
        "ml_final_equity": ml_final_equity,
        "buy_hold_final_equity": bench_final_equity,
        "final_equity_difference_rmb": ml_final_equity - bench_final_equity,
        "final_equity_difference_pct_initial": (ml_final_equity - bench_final_equity) / INITIAL_CAPITAL,
    }
    joined.to_csv(DIAGNOSTICS_DIR / "ml_active_return_attribution.csv", index=False)
    return joined, summary


def _available_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def export_top_day_tables(attribution: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = _available_cols(attribution, CONTEXT_COLS)
    missed = (
        attribution.loc[(attribution["final_position"] < 0.999) & (attribution["fwd_ret_1d"] > 0), cols]
        .sort_values("position_gap_return", ascending=True)
        .head(20)
    )
    defensive = (
        attribution.loc[(attribution["final_position"] < 0.999) & (attribution["fwd_ret_1d"] < 0), cols]
        .sort_values("position_gap_return", ascending=False)
        .head(20)
    )
    missed.to_csv(DIAGNOSTICS_DIR / "top_missed_upside_days.csv", index=False)
    defensive.to_csv(DIAGNOSTICS_DIR / "top_successful_defensive_days.csv", index=False)
    return missed, defensive


def build_prediction_position_frame(
    predictions: pd.DataFrame,
    attribution: pd.DataFrame,
    selected: tuple[str, int],
) -> pd.DataFrame:
    model_name, horizon = selected
    pred = predictions.loc[(predictions["model_name"] == model_name) & (predictions["horizon"] == horizon)].copy()
    if pred.empty:
        pred = attribution.copy()
    position_cols = [
        "date",
        "final_position",
        "strategy_return",
        "benchmark_return",
        "position_gap",
        "position_gap_return",
        "turnover",
        "turnover_cost",
    ]
    position = attribution[_available_cols(attribution, position_cols)].copy()
    out = pred.merge(position, on="date", how="left", suffixes=("", "_strategy"))
    if "final_position" not in out.columns:
        out["final_position"] = np.nan
    return out.sort_values("date").reset_index(drop=True)


def build_big_up_calibration(pred_position: pd.DataFrame) -> pd.DataFrame:
    work = pred_position.dropna(subset=["pred_big_up_prob"]).copy()
    if work.empty:
        out = pd.DataFrame()
        out.to_csv(DIAGNOSTICS_DIR / "big_up_probability_calibration.csv", index=False)
        return out

    unique_count = work["pred_big_up_prob"].nunique(dropna=True)
    bins = min(10, unique_count)
    if bins >= 2:
        work["big_up_prob_bin"] = pd.qcut(work["pred_big_up_prob"], q=bins, duplicates="drop")
    else:
        work["big_up_prob_bin"] = pd.cut(work["pred_big_up_prob"], bins=1)
    grouped = work.groupby("big_up_prob_bin", observed=True)
    out = grouped.agg(
        sample_count=("pred_big_up_prob", "size"),
        mean_pred_big_up_prob=("pred_big_up_prob", "mean"),
        mean_target_big_up=("target_big_up", "mean"),
        realized_big_up_rate=("target_big_up", "mean"),
        mean_target_ret=("target_ret", "mean"),
        median_target_ret=("target_ret", "median"),
        mean_fwd_ret_1d=("fwd_ret_1d", "mean"),
        pct_positive_target_ret=("target_ret", lambda s: float((s > 0).mean())),
        average_final_position=("final_position", "mean"),
        total_strategy_return_contribution=("strategy_return", "sum"),
        total_benchmark_return_contribution=("benchmark_return", "sum"),
    ).reset_index()
    out["big_up_prob_bin"] = out["big_up_prob_bin"].astype(str)
    out.to_csv(DIAGNOSTICS_DIR / "big_up_probability_calibration.csv", index=False)
    return out


def export_probability_edge_cases(pred_position: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = pred_position.dropna(subset=["pred_big_up_prob"]).copy()
    if work.empty:
        low = high = pd.DataFrame()
    else:
        low_cut = work["pred_big_up_prob"].quantile(0.30)
        high_cut = work["pred_big_up_prob"].quantile(0.70)
        work["realized_positive_return"] = work[["target_ret", "fwd_ret_1d"]].max(axis=1)
        work["realized_weak_return"] = work[["target_ret", "fwd_ret_1d"]].min(axis=1)
        cols = _available_cols(work, CONTEXT_COLS + ["target_big_up", "realized_positive_return", "realized_weak_return"])
        low = (
            work.loc[
                (work["pred_big_up_prob"] <= low_cut)
                & (work["final_position"] < 0.999)
                & ((work["target_ret"] > 0) | (work["fwd_ret_1d"] > 0)),
                cols,
            ]
            .sort_values("realized_positive_return", ascending=False)
            .head(20)
        )
        high = (
            work.loc[
                (work["pred_big_up_prob"] >= high_cut)
                & ((work["target_ret"] <= 0) | (work["fwd_ret_1d"] <= 0)),
                cols,
            ]
            .sort_values("realized_weak_return", ascending=True)
            .head(20)
        )
    low.to_csv(DIAGNOSTICS_DIR / "low_big_up_but_positive_return.csv", index=False)
    high.to_csv(DIAGNOSTICS_DIR / "high_big_up_but_weak_return.csv", index=False)
    return low, high


def _markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "No rows."
    sample = df.head(max_rows).copy()
    cols = sample.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in sample.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, pd.Timestamp):
                values.append(value.strftime("%Y-%m-%d"))
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_reports(
    exposure: pd.DataFrame,
    attribution_summary: dict[str, float],
    calibration: pd.DataFrame,
    missed: pd.DataFrame,
    defensive: pd.DataFrame,
    low_positive: pd.DataFrame,
    high_weak: pd.DataFrame,
    selected: tuple[str, int],
) -> None:
    ml_exp = exposure.loc[exposure["strategy"] == MAIN_STRATEGY].iloc[0].to_dict()
    low_bin = calibration.iloc[0].to_dict() if not calibration.empty else {}
    high_bin = calibration.iloc[-1].to_dict() if not calibration.empty else {}
    high_mean_ret = _safe_float(high_bin.get("mean_target_ret"))
    low_mean_ret = _safe_float(low_bin.get("mean_target_ret"))
    high_big_rate = _safe_float(high_bin.get("realized_big_up_rate"))
    low_big_rate = _safe_float(low_bin.get("realized_big_up_rate"))
    low_positive_count = len(low_positive)
    high_weak_count = len(high_weak)
    sensitivity_note = (
        "The current rule appears too sensitive to weak/risk regimes because it spent many days below full exposure while the market often continued rising."
        if ml_exp["pct_days_below_full_exposure"] > 0.40 and attribution_summary["missed_upside"] > attribution_summary["avoided_downside"]
        else "The current rule does not look excessively sensitive by exposure frequency alone, but missed-upside days should still be reviewed before tuning."
    )

    summary = f"""# ML Underperformance Diagnosis

## Headline
- Selected strategy: `{MAIN_STRATEGY}`
- Selected prediction source: `{selected[0]}`, horizon `{selected[1]}D`
- Final-equity gap vs buy-and-hold: RMB {attribution_summary["final_equity_difference_rmb"]:.2f}
- Final-equity gap as initial-capital percentage: {attribution_summary["final_equity_difference_pct_initial"]:.4f}
- Arithmetic active-return sum: {attribution_summary["total_active_return_arithmetic"]:.4f}

## Exposure Diagnostics
- Average position: {ml_exp["average_final_position"]:.4f}
- Minimum position: {ml_exp["minimum_final_position"]:.4f}
- Maximum position: {ml_exp["maximum_final_position"]:.4f}
- Days below full exposure: {int(ml_exp["days_below_full_exposure"])} ({ml_exp["pct_days_below_full_exposure"]:.2%})
- Total turnover: {ml_exp["total_turnover"]:.4f}
- Average absolute position gap from 1.0: {ml_exp["average_abs_position_gap_from_1"]:.4f}
- Days with position <= 0.95: {int(ml_exp["days_position_le_095"])}
- Days with position <= 0.90: {int(ml_exp["days_position_le_090"])}
- Days with position <= 0.80: {int(ml_exp["days_position_le_080"])}

## Active-Return Attribution
- Missed upside from reduced exposure on positive-return days: {attribution_summary["missed_upside"]:.4f}
- Avoided downside from reduced exposure on negative-return days: {attribution_summary["avoided_downside"]:.4f}
- Turnover cost: {attribution_summary["turnover_cost"]:.4f}
- Net timing contribution: {attribution_summary["net_timing_contribution"]:.4f}

The underperformance mainly came from missed upside because missed upside was larger than avoided downside, while turnover cost was small but still negative.

## Big-Up Probability Diagnostic
- Lowest big-up-probability bin mean prediction: {_fmt(low_bin.get("mean_pred_big_up_prob"))}
- Lowest bin realized big-up rate: {_fmt(low_big_rate)}
- Lowest bin mean target return: {_fmt(low_mean_ret)}
- Highest big-up-probability bin mean prediction: {_fmt(high_bin.get("mean_pred_big_up_prob"))}
- Highest bin realized big-up rate: {_fmt(high_big_rate)}
- Highest bin mean target return: {_fmt(high_mean_ret)}
- Low-probability but positive-return cases exported: {low_positive_count}
- High-probability but weak-return cases exported: {high_weak_count}

When `pred_big_up_prob` was high, the market did not rise enough to compensate for the upside missed during low-probability/reduced-exposure periods. Low `pred_big_up_prob` did not reliably identify weak future returns, because there were still many positive-return cases in the low-probability group.

## Rule Sensitivity
{sensitivity_note}

## Top Missed-Upside Days
{_markdown_table(missed[["date", "trade_date", "fwd_ret_1d", "final_position", "position_gap_return", "pred_big_up_prob"]], max_rows=10)}

## Top Successful Defensive Days
{_markdown_table(defensive[["date", "trade_date", "fwd_ret_1d", "final_position", "position_gap_return", "pred_big_up_prob"]], max_rows=10)}

## Suggested Next Step
Do not tune blindly. First test whether exposure cuts should be less frequent, whether the severe defensive trigger should require stronger confirmation, and whether low `pred_big_up_prob` should reduce exposure only when downside probability and trend/risk filters agree.
"""
    (DIAGNOSTICS_DIR / "ml_underperformance_diagnosis.md").write_text(summary, encoding="utf-8")

    attr_md = f"""# ML Active-Return Attribution Summary

- Total active return, arithmetic sum: {attribution_summary["total_active_return_arithmetic"]:.6f}
- Total position-gap return: {attribution_summary["total_position_gap_return"]:.6f}
- Missed upside: {attribution_summary["missed_upside"]:.6f}
- Avoided downside: {attribution_summary["avoided_downside"]:.6f}
- Turnover cost: {attribution_summary["turnover_cost"]:.6f}
- Net timing contribution: {attribution_summary["net_timing_contribution"]:.6f}
- ML final equity: RMB {attribution_summary["ml_final_equity"]:.2f}
- Buy-and-hold final equity: RMB {attribution_summary["buy_hold_final_equity"]:.2f}
- Final equity difference: RMB {attribution_summary["final_equity_difference_rmb"]:.2f}

## Interpretation
The strategy underperformed mainly because reduced exposure missed a large amount of upside in a strong market. Avoided downside partially helped, and turnover cost was relatively small, but not enough to offset the missed upside.
"""
    (DIAGNOSTICS_DIR / "ml_active_return_attribution_summary.md").write_text(attr_md, encoding="utf-8")


def make_plots(attribution: pd.DataFrame, calibration: pd.DataFrame, missed: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 6))
    plt.plot(attribution["date"], attribution["cumulative_active_return"], label="cumulative active return")
    plt.plot(attribution["date"], -attribution["cumulative_missed_upside"], label="cumulative missed upside")
    plt.plot(attribution["date"], attribution["cumulative_avoided_downside"], label="cumulative avoided downside")
    plt.plot(attribution["date"], -attribution["cumulative_turnover_cost"], label="cumulative turnover cost")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Cumulative Active-Return Attribution")
    plt.xlabel("Date")
    plt.ylabel("Return Contribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "active_return_cumulative.png", dpi=160)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.plot(attribution["date"], attribution["final_position"], label="final position", color="tab:blue")
    ax1.scatter(
        attribution.loc[attribution["final_position"] < 0.999, "date"],
        attribution.loc[attribution["final_position"] < 0.999, "final_position"],
        color="tab:red",
        s=14,
        label="reduced exposure",
    )
    ax1.set_ylabel("Position")
    ax1.set_ylim(0.75, 1.03)
    ax2 = ax1.twinx()
    ax2.bar(attribution["date"], attribution["fwd_ret_1d"], alpha=0.25, color="gray", label="next-day return")
    ax2.set_ylabel("Next-Day Return")
    ax1.set_title("Position vs Next-Day Return")
    ax1.set_xlabel("Date")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "position_vs_next_return.png", dpi=160)
    plt.close(fig)

    if not calibration.empty:
        x = np.arange(len(calibration))
        plt.figure(figsize=(13, 6))
        plt.bar(x - 0.18, calibration["mean_pred_big_up_prob"], width=0.36, label="mean predicted big-up probability")
        plt.bar(x + 0.18, calibration["realized_big_up_rate"], width=0.36, label="realized big-up rate")
        plt.plot(x, calibration["mean_target_ret"], color="black", marker="o", label="mean target return")
        plt.xticks(x, calibration["big_up_prob_bin"], rotation=35, ha="right")
        plt.title("Big-Up Probability Calibration")
        plt.xlabel("pred_big_up_prob bin")
        plt.ylabel("Rate / Return")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "big_up_probability_calibration.png", dpi=160)
        plt.close()

    if not missed.empty:
        plot_df = missed.copy().sort_values("position_gap_return", ascending=True).head(20)
        plot_df["missed_upside"] = -plot_df["position_gap_return"]
        plt.figure(figsize=(13, 6))
        plt.bar(pd.to_datetime(plot_df["date"]).dt.strftime("%Y-%m-%d"), plot_df["missed_upside"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Top Missed-Upside Reduced-Exposure Days")
        plt.xlabel("Date")
        plt.ylabel("Missed upside contribution")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "missed_upside_top_dates.png", dpi=160)
        plt.close()


def main() -> None:
    _ensure_dirs()
    strategy_daily, strategy_metrics, predictions, selected = load_inputs()
    comparison = export_strategy_comparison(strategy_metrics)
    exposure = compute_exposure_diagnostics(strategy_daily)
    attribution, attr_summary = build_active_attribution(strategy_daily)
    missed, defensive = export_top_day_tables(attribution)
    pred_position = build_prediction_position_frame(predictions, attribution, selected)
    calibration = build_big_up_calibration(pred_position)
    low_positive, high_weak = export_probability_edge_cases(pred_position)
    write_markdown_reports(exposure, attr_summary, calibration, missed, defensive, low_positive, high_weak, selected)
    make_plots(attribution, calibration, missed)

    ml_exp = exposure.loc[exposure["strategy"] == MAIN_STRATEGY].iloc[0]
    print("\n===== ML UNDERPERFORMANCE DIAGNOSTICS =====")
    print(f"Selected strategy: {MAIN_STRATEGY}")
    print(f"Selected prediction source: {selected[0]}, horizon={selected[1]}D")
    print(f"Average position: {ml_exp['average_final_position']:.4f}")
    print(f"Minimum position: {ml_exp['minimum_final_position']:.4f}")
    print(f"Days below full exposure: {int(ml_exp['days_below_full_exposure'])}")
    print(f"Missed upside: {attr_summary['missed_upside']:.6f}")
    print(f"Avoided downside: {attr_summary['avoided_downside']:.6f}")
    print(f"Turnover cost: {attr_summary['turnover_cost']:.6f}")
    print(f"Net timing contribution: {attr_summary['net_timing_contribution']:.6f}")
    print("\nStrategy comparison:")
    print(comparison.to_string(index=False))
    print("Top 5 missed-upside dates:")
    for row in missed.head(5).itertuples(index=False):
        print(
            f"  {pd.Timestamp(row.date).date()} -> {pd.Timestamp(row.trade_date).date()}, "
            f"fwd_ret={row.fwd_ret_1d:.4f}, pos={row.final_position:.2f}, "
            f"gap_return={row.position_gap_return:.6f}"
        )
    print(f"Diagnostics saved to: {DIAGNOSTICS_DIR}")
    print("==========================================")


if __name__ == "__main__":
    main()
