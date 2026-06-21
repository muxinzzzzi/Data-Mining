from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "current" / "plots"

MPL_CACHE_DIR = Path("/private/tmp/matplotlib-final-report-cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib
import matplotlib.dates as mdates

matplotlib.use("Agg")

import matplotlib.pyplot as plt

DAILY_PATH = OUTPUTS_DIR / "strategy_daily_all.csv"
METRICS_PATH = OUTPUTS_DIR / "strategy_metrics_all.csv"
EVENT_CSV_PATH = OUTPUTS_DIR / "drawdown_event_diagnostics.csv"
SIGNAL_QUANTILE_PATH = OUTPUTS_DIR / "signal_quantile_stability.csv"
SIGNAL_QUANTILE_SUMMARY_PATH = OUTPUTS_DIR / "signal_quantile_stability_summary.csv"
SIGNAL_BUCKET_PATH = OUTPUTS_DIR / "signal_bucket_diagnostics.csv"
MAIN_ATTRIBUTION_PATH = OUTPUTS_DIR / "main_strategy_attribution.csv"
EVENT_TIMELINE_SOURCE = OUTPUTS_DIR / "plots" / "drawdown_event_risk_signal_timeline.png"


REQUESTED_EQUITY_STRATEGIES = [
    "buy_hold",
    "ml_stability_aware_high_participation",
    "ml_early_stress_risk_budget",
    "ml_index_enhanced_plus_120",
]

REQUESTED_DRAWDOWN_STRATEGIES = [
    "buy_hold",
    "ml_stability_aware_high_participation",
    "ml_early_stress_risk_budget",
]

POSITION_STRATEGIES = [
    "ml_stability_aware_high_participation",
    "ml_early_stress_risk_budget",
]

STRATEGY_ALIASES = {
    "ml_index_enhanced_plus_120": ["ml_big_up_plus_120"],
}

DISPLAY_NAMES = {
    "buy_hold": "Buy & Hold",
    "ml_stability_aware_high_participation": "Stability-Aware High Participation",
    "ml_early_stress_risk_budget": "Early-Stress Risk Budget",
    "ml_index_enhanced_plus_120": "Enhanced Exposure +120",
    "ml_big_up_plus_120": "Enhanced Exposure +120",
    "ml_risk_budget_enhancement": "Risk-Budget Enhancement",
    "ml_upside_participation_enhancement": "Upside Participation",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_required_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {DAILY_PATH}")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Required file not found: {METRICS_PATH}")
    daily = pd.read_csv(DAILY_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    return daily, metrics


def display_name(strategy: str) -> str:
    return DISPLAY_NAMES.get(strategy, strategy)


def resolve_strategy(requested: str, available: set[str]) -> str | None:
    if requested in available:
        return requested
    for alias in STRATEGY_ALIASES.get(requested, []):
        if alias in available:
            print(f"Strategy {requested} not found; using available alias {alias}.")
            return alias
    print(f"Strategy {requested} not found; skipping it.")
    return None


def resolve_strategy_list(requested: list[str], available: set[str]) -> list[str]:
    resolved: list[str] = []
    for strategy in requested:
        actual = resolve_strategy(strategy, available)
        if actual and actual not in resolved:
            resolved.append(actual)
    return resolved


def save_figure(fig: plt.Figure, filename: str, generated: list[tuple[str, str]]) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {path}")
    generated.append((filename, str(path)))
    return path


def format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_horizontalalignment("right")


def strategy_frame(daily: pd.DataFrame, strategy: str) -> pd.DataFrame:
    frame = daily.loc[daily["strategy"] == strategy].copy().sort_values("date")
    return frame.dropna(subset=["date"])


def plot_final_equity_curve(daily: pd.DataFrame, generated: list[tuple[str, str]]) -> None:
    strategies = resolve_strategy_list(REQUESTED_EQUITY_STRATEGIES, set(daily["strategy"].dropna()))
    if not strategies:
        print("No requested strategies available for final_equity_curve_comparison.png; skipping.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy in strategies:
        frame = strategy_frame(daily, strategy)
        if frame.empty or "equity" not in frame.columns:
            print(f"Strategy {strategy} has no equity data; skipping in equity curve.")
            continue
        base = float(frame["equity"].dropna().iloc[0]) if frame["equity"].notna().any() else np.nan
        y = frame["equity"].astype(float) / base if np.isfinite(base) and base != 0 else frame["equity"].astype(float)
        ax.plot(frame["date"], y, linewidth=1.8, label=display_name(strategy))
    ax.set_title("Final Equity Curve Comparison")
    ax.set_ylabel("Equity Growth (Start = 1.0)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    format_date_axis(ax)
    save_figure(fig, "final_equity_curve_comparison.png", generated)


def plot_final_drawdown(daily: pd.DataFrame, generated: list[tuple[str, str]]) -> None:
    strategies = resolve_strategy_list(REQUESTED_DRAWDOWN_STRATEGIES, set(daily["strategy"].dropna()))
    if not strategies:
        print("No requested strategies available for final_drawdown_comparison.png; skipping.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy in strategies:
        frame = strategy_frame(daily, strategy)
        if frame.empty:
            continue
        drawdown_col = "buy_hold_drawdown" if strategy == "buy_hold" and "buy_hold_drawdown" in frame.columns else "drawdown"
        if drawdown_col not in frame.columns:
            print(f"Strategy {strategy} has no drawdown data; skipping in drawdown curve.")
            continue
        ax.plot(frame["date"], frame[drawdown_col].astype(float), linewidth=1.8, label=display_name(strategy))
    ax.set_title("Final Drawdown Comparison")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    format_date_axis(ax)
    save_figure(fig, "final_drawdown_comparison.png", generated)


def plot_metrics_bar(metrics: pd.DataFrame, generated: list[tuple[str, str]]) -> None:
    strategies = resolve_strategy_list(REQUESTED_EQUITY_STRATEGIES, set(metrics["strategy"].dropna()))
    rows = metrics.loc[metrics["strategy"].isin(strategies)].copy()
    if rows.empty:
        print("No requested strategies available for final_strategy_metrics_bar.png; skipping.")
        return
    rows["strategy_label"] = rows["strategy"].map(display_name)
    rows["max_drawdown_magnitude"] = rows["max_drawdown"].astype(float).abs()
    metric_specs = [
        ("total_return", "Total Return"),
        ("max_drawdown_magnitude", "Maximum Drawdown Magnitude"),
        ("sharpe", "Sharpe Ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (col, title) in zip(axes, metric_specs):
        use = rows.set_index("strategy_label").loc[:, col].astype(float)
        colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"][: len(use)]
        ax.bar(use.index, use.values, color=colors)
        ax.set_title(title)
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
    save_figure(fig, "final_strategy_metrics_bar.png", generated)


def plot_position_exposure(daily: pd.DataFrame, generated: list[tuple[str, str]]) -> None:
    strategies = resolve_strategy_list(POSITION_STRATEGIES, set(daily["strategy"].dropna()))
    if not strategies:
        print("No requested strategies available for final_position_exposure.png; skipping.")
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for strategy in strategies:
        frame = strategy_frame(daily, strategy)
        if frame.empty or "final_position" not in frame.columns:
            print(f"Strategy {strategy} has no final_position data; skipping exposure plot.")
            continue
        ax.plot(frame["date"], frame["final_position"].astype(float), linewidth=1.6, label=display_name(strategy))
    ax.set_title("Final Position Exposure")
    ax.set_ylabel("Final Position")
    ax.set_xlabel("Date")
    ax.set_ylim(0.75, 1.03)
    ax.legend(loc="best")
    format_date_axis(ax)
    save_figure(fig, "final_position_exposure.png", generated)


def plot_drawdown_event_diagnostics(generated: list[tuple[str, str]]) -> None:
    target = PLOTS_DIR / "final_drawdown_event_diagnostics.png"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if EVENT_TIMELINE_SOURCE.exists():
        shutil.copy2(EVENT_TIMELINE_SOURCE, target)
        print(f"Copied figure: {EVENT_TIMELINE_SOURCE} -> {target}")
        generated.append(("final_drawdown_event_diagnostics.png", str(target)))
        return
    if not EVENT_CSV_PATH.exists():
        print("No drawdown event plot or diagnostics CSV found; skipping final_drawdown_event_diagnostics.png.")
        return
    event = pd.read_csv(EVENT_CSV_PATH)
    if event.empty or "date" not in event.columns or "strategy" not in event.columns:
        print("Drawdown event diagnostics CSV is missing date/strategy columns; skipping.")
        return
    event["date"] = pd.to_datetime(event["date"], errors="coerce")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for strategy, col, label in [
        ("buy_hold", "drawdown", "Buy & Hold Drawdown"),
        ("ml_risk_budget_enhancement", "final_position", "Risk-Budget Position"),
        ("ml_early_stress_risk_budget", "final_position", "Early-Stress Position"),
    ]:
        frame = event.loc[event["strategy"] == strategy].sort_values("date")
        if frame.empty or col not in frame.columns:
            print(f"Event diagnostics missing {strategy}/{col}; partial event plot will omit it.")
            continue
        ax = axes[0] if col == "drawdown" else axes[1 if strategy == "ml_risk_budget_enhancement" else 2]
        ax.plot(frame["date"], frame[col].astype(float), label=label, linewidth=1.7)
        ax.legend(loc="best")
    axes[0].set_title("Drawdown Event Diagnostics")
    axes[0].set_ylabel("Drawdown")
    axes[1].set_ylabel("Risk-Budget Position")
    axes[2].set_ylabel("Early-Stress Position")
    axes[2].set_xlabel("Date")
    format_date_axis(axes[2])
    save_figure(fig, "final_drawdown_event_diagnostics.png", generated)


def plot_signal_quantile_stability(generated: list[tuple[str, str]]) -> None:
    if SIGNAL_QUANTILE_SUMMARY_PATH.exists():
        summary = pd.read_csv(SIGNAL_QUANTILE_SUMMARY_PATH)
    elif SIGNAL_QUANTILE_PATH.exists():
        quantile = pd.read_csv(SIGNAL_QUANTILE_PATH)
        required = {"period", "signal_name", "quantile_index", "mean_forward_return"}
        if not required.issubset(quantile.columns):
            print("Signal quantile CSV lacks required columns; skipping signal_quantile_stability.png.")
            return
        summary = (
            quantile.loc[quantile["quantile_index"].isin([1, 5])]
            .pivot_table(
                index=["period", "model_name", "horizon", "signal_name"],
                columns="quantile_index",
                values="mean_forward_return",
                aggfunc="mean",
            )
            .reset_index()
        )
        summary["bottom_group_mean_return"] = summary.get(1)
        summary["top_group_mean_return"] = summary.get(5)
        summary["top_bottom_spread"] = summary["top_group_mean_return"] - summary["bottom_group_mean_return"]
    else:
        print("No signal quantile files found; skipping signal_quantile_stability.png.")
        return
    required = {"period", "signal_name", "top_bottom_spread"}
    if summary.empty or not required.issubset(summary.columns):
        print("Signal quantile summary lacks required columns; skipping signal_quantile_stability.png.")
        return
    use = (
        summary.loc[summary["period"].isin(["validation", "test"])]
        .groupby(["period", "signal_name"], as_index=False)["top_bottom_spread"]
        .mean()
    )
    if use.empty:
        print("No validation/test signal quantile rows found; skipping signal_quantile_stability.png.")
        return
    pivot = use.pivot(index="signal_name", columns="period", values="top_bottom_spread").fillna(0.0)
    ordered = [col for col in ["validation", "test"] if col in pivot.columns]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.36 if len(ordered) > 1 else 0.5
    for idx, period in enumerate(ordered):
        offset = (idx - (len(ordered) - 1) / 2) * width
        ax.bar(x + offset, pivot[period].values, width=width, label=period.title())
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Signal Quantile Stability: Top-Bottom Forward Return Spread")
    ax.set_ylabel("Mean Forward Return Spread (Q5 - Q1)")
    ax.set_xlabel("Signal")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.legend(loc="best")
    save_figure(fig, "signal_quantile_stability.png", generated)


def plot_bigup_trade_signal_diagnostics(generated: list[tuple[str, str]]) -> None:
    if SIGNAL_QUANTILE_PATH.exists():
        quantile = pd.read_csv(SIGNAL_QUANTILE_PATH)
        signal_col = "signal_name"
        return_col = "mean_forward_return"
        group_col = "quantile_index"
    elif SIGNAL_BUCKET_PATH.exists():
        quantile = pd.read_csv(SIGNAL_BUCKET_PATH)
        signal_col = "signal"
        return_col = "future_1d_return_mean"
        group_col = "bucket"
    else:
        print("No signal bucket or quantile file found; skipping bigup_trade_signal_diagnostics.png.")
        return
    needed = {"period", signal_col, group_col, return_col}
    if quantile.empty or not needed.issubset(quantile.columns):
        print("Signal bucket/quantile file lacks required columns; skipping bigup_trade_signal_diagnostics.png.")
        return
    signals = ["pred_big_up_prob", "pred_trade_prob"]
    use = quantile.loc[
        quantile[signal_col].isin(signals) & quantile["period"].isin(["validation", "test"])
    ].copy()
    if use.empty:
        print("No pred_big_up_prob or pred_trade_prob rows found; skipping bigup_trade_signal_diagnostics.png.")
        return
    agg = use.groupby(["period", signal_col, group_col], as_index=False)[return_col].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, signal in zip(axes, signals):
        sframe = agg.loc[agg[signal_col] == signal].sort_values(group_col)
        if sframe.empty:
            ax.set_visible(False)
            print(f"No rows for {signal}; omitting it from bigup/trade signal diagnostics.")
            continue
        for period, pframe in sframe.groupby("period"):
            ax.plot(
                pframe[group_col].astype(float),
                pframe[return_col].astype(float),
                marker="o",
                linewidth=1.8,
                label=period.title(),
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(signal)
        ax.set_xlabel("Quantile Group")
        ax.set_ylabel("Mean Forward Return")
        ax.legend(loc="best")
    fig.suptitle("Big-Up and Tradeable-Up Signal Diagnostics", y=1.02)
    save_figure(fig, "bigup_trade_signal_diagnostics.png", generated)


def plot_timing_attribution(generated: list[tuple[str, str]], metrics: pd.DataFrame) -> None:
    rows: list[dict[str, float | str]] = []
    if MAIN_ATTRIBUTION_PATH.exists():
        attr = pd.read_csv(MAIN_ATTRIBUTION_PATH)
        required = {"strategy", "missed_upside", "avoided_downside", "transaction_cost", "net_position_timing_contribution"}
        if required.issubset(attr.columns) and not attr.empty:
            rows.append(
                {
                    "strategy": str(attr["strategy"].dropna().iloc[0]),
                    "missed_upside": float(attr["missed_upside"].sum()),
                    "avoided_downside": float(attr["avoided_downside"].sum()),
                    "transaction_cost": float(attr["transaction_cost"].sum()),
                    "net_timing_contribution": float(attr["net_position_timing_contribution"].sum()),
                }
            )
    metric_required = {"strategy", "missed_upside", "avoided_downside", "total_transaction_cost", "net_timing_contribution"}
    if metric_required.issubset(metrics.columns):
        for strategy in [
            "ml_stability_aware_high_participation",
            "ml_early_stress_risk_budget",
            "ml_risk_budget_enhancement",
            "ml_upside_participation_enhancement",
        ]:
            match = metrics.loc[metrics["strategy"] == strategy]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                {
                    "strategy": strategy,
                    "missed_upside": float(row["missed_upside"]),
                    "avoided_downside": float(row["avoided_downside"]),
                    "transaction_cost": float(row["total_transaction_cost"]),
                    "net_timing_contribution": float(row["net_timing_contribution"]),
                }
            )
    if not rows:
        print("No structured attribution CSV/data found; skipping timing_attribution_bar.png.")
        return
    data = pd.DataFrame(rows).drop_duplicates(subset=["strategy"], keep="last")
    metrics_to_plot = ["missed_upside", "avoided_downside", "transaction_cost", "net_timing_contribution"]
    x = np.arange(len(data))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, metric in enumerate(metrics_to_plot):
        ax.bar(x + (idx - 1.5) * width, data[metric].astype(float), width=width, label=metric)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Timing Attribution")
    ax.set_ylabel("Contribution")
    ax.set_xlabel("Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(s) for s in data["strategy"]], rotation=30, ha="right")
    ax.legend(loc="best", ncol=2)
    save_figure(fig, "timing_attribution_bar.png", generated)


def write_figure_index(generated: list[tuple[str, str]]) -> None:
    descriptions = {
        "final_equity_curve_comparison.png": (
            "Strategy Results",
            "Compares cumulative equity growth for buy-and-hold, high-participation ML strategies, and enhanced exposure when available.",
        ),
        "final_drawdown_comparison.png": (
            "Risk and Drawdown",
            "Compares drawdown paths for buy-and-hold, stability-aware high participation, and early-stress risk budget.",
        ),
        "final_strategy_metrics_bar.png": (
            "Strategy Results",
            "Shows total return, maximum drawdown magnitude, and Sharpe ratio side by side.",
        ),
        "final_position_exposure.png": (
            "Strategy Design",
            "Shows that the ML strategies are high-participation index enhancement strategies, not 0/1 trading systems.",
        ),
        "final_drawdown_event_diagnostics.png": (
            "Drawdown Event Diagnostics",
            "Highlights what happened during the benchmark maximum drawdown event and whether risk signals/positions reacted.",
        ),
        "signal_quantile_stability.png": (
            "Signal Validation",
            "Summarizes validation/test top-minus-bottom quantile forward-return spread by signal.",
        ),
        "bigup_trade_signal_diagnostics.png": (
            "Signal Validation",
            "Shows quantile-level forward returns for pred_big_up_prob and pred_trade_prob.",
        ),
        "timing_attribution_bar.png": (
            "Attribution",
            "Breaks timing contribution into missed upside, avoided downside, transaction cost, and net contribution.",
        ),
    }
    generated_names = {name for name, _ in generated}
    lines = [
        "# Final Report Figure Index",
        "",
        "This index is generated by `scripts/plot_final_report_figures.py`. The script only reads existing outputs and writes report figures.",
        "",
        "| Figure | Suggested Report Section | What It Shows | Status |",
        "| --- | --- | --- | --- |",
    ]
    for filename, (section, description) in descriptions.items():
        status = "Generated" if filename in generated_names else "Skipped"
        lines.append(f"| `{filename}` | {section} | {description} | {status} |")
    path = PLOTS_DIR / "FIGURE_INDEX.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved figure index: {path}")


def main() -> None:
    configure_matplotlib()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    daily, metrics = load_required_data()
    generated: list[tuple[str, str]] = []

    plot_final_equity_curve(daily, generated)
    plot_final_drawdown(daily, generated)
    plot_metrics_bar(metrics, generated)
    plot_position_exposure(daily, generated)
    plot_drawdown_event_diagnostics(generated)
    plot_signal_quantile_stability(generated)
    plot_bigup_trade_signal_diagnostics(generated)
    plot_timing_attribution(generated, metrics)
    write_figure_index(generated)

    print("===== FINAL REPORT FIGURES COMPLETE =====")
    print(f"Generated figures: {len(generated)}")
    for _, path in generated:
        print(f"- {path}")
    print(f"Figure index: {PLOTS_DIR / 'FIGURE_INDEX.md'}")
    print("=========================================")


if __name__ == "__main__":
    main()
