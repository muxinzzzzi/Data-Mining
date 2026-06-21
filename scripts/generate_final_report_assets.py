from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = PROJECT_ROOT / "outputs" / "current"
FIGURE_DIR = CURRENT_DIR / "report_figures"
TABLE_DIR = CURRENT_DIR / "report_tables"

MPL_CACHE_DIR = Path("/private/tmp/matplotlib-report-assets-cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")

import matplotlib.pyplot as plt


STRATEGY_DISPLAY = {
    "buy_hold": "Buy and Hold",
    "ml_risk_budget_enhancement": "Risk Budget",
    "ml_early_stress_risk_budget": "Early Stress",
    "ml_external_soft_lite_risk_budget_index_only": "External Soft-Lite",
}

MODEL_DISPLAY = {
    "mlp_small_fs30": "MLP",
    "LightGBM": "LightGBM",
    "RandomForest": "RandomForest",
    "LogisticRegression": "LogisticRegression",
    "RidgeClassifier": "RidgeClassifier",
}

REPRESENTATIVE_MODELS = [
    ("MLP", "mlp_small_fs30", 20, "Validation-best prediction model"),
    ("LightGBM", "LightGBM", 5, "Main risk-budget strategy model"),
    ("RandomForest", "RandomForest", 5, "Early-stress strategy model"),
    ("LogisticRegression", "LogisticRegression", 5, "Linear baseline"),
    ("RidgeClassifier", "RidgeClassifier", 5, "Linear stability baseline"),
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
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


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    path = CURRENT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required current output is missing: {path}")
    return pd.read_csv(path)


def save_png(fig: plt.Figure, filename: str) -> Path:
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    path = TABLE_DIR / filename
    df.to_csv(path, index=False)
    return path


def format_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_horizontalalignment("right")


def display_strategy(strategy: str) -> str:
    return STRATEGY_DISPLAY.get(strategy, strategy)


def strategy_frame(daily: pd.DataFrame, strategy: str) -> pd.DataFrame:
    frame = daily.loc[daily["strategy"] == strategy].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date")


def representative_model_scores(stability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, model_name, horizon, role in REPRESENTATIVE_MODELS:
        subset = stability.loc[
            (stability["model_name"] == model_name)
            & (stability["horizon"].astype(int) == horizon)
        ]
        if subset.empty:
            rows.append(
                {
                    "model": label,
                    "source_model_name": model_name,
                    "horizon": horizon,
                    "role": role,
                    "validation_score": np.nan,
                    "test_score": np.nan,
                    "score_gap": np.nan,
                    "source_status": "missing",
                }
            )
            continue
        row = subset.iloc[0]
        validation_score = float(row["validation_selection_score"])
        test_score = float(row["test_selection_score"])
        rows.append(
            {
                "model": label,
                "source_model_name": model_name,
                "horizon": horizon,
                "role": role,
                "validation_score": validation_score,
                "test_score": test_score,
                "score_gap": validation_score - test_score,
                "source_status": "available",
            }
        )
    return pd.DataFrame(rows)


def representative_prediction_auc(validation: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, model_name, horizon, role in REPRESENTATIVE_MODELS:
        val = validation.loc[
            (validation["model_name"] == model_name) & (validation["horizon"].astype(int) == horizon)
        ]
        tst = test.loc[
            (test["model_name"] == model_name) & (test["horizon"].astype(int) == horizon)
        ]
        row = {
            "model": label,
            "source_model_name": model_name,
            "horizon": horizon,
            "role": role,
        }
        for prefix, frame in [("validation", val), ("test", tst)]:
            if frame.empty:
                row.update(
                    {
                        f"{prefix}_ordinary_direction_auc": np.nan,
                        f"{prefix}_clean_direction_auc": np.nan,
                        f"{prefix}_big_up_auc": np.nan,
                        f"{prefix}_selection_score": np.nan,
                    }
                )
                continue
            item = frame.iloc[0]
            row.update(
                {
                    f"{prefix}_ordinary_direction_auc": float(item["AUC"]),
                    f"{prefix}_clean_direction_auc": float(item["clean_direction_AUC"]),
                    f"{prefix}_big_up_auc": float(item["big_up_AUC"]),
                    f"{prefix}_selection_score": float(item["selection_score"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_equity_curve(daily: pd.DataFrame) -> Path:
    strategies = [
        "buy_hold",
        "ml_risk_budget_enhancement",
        "ml_early_stress_risk_budget",
    ]
    colors = {
        "buy_hold": "#4C78A8",
        "ml_risk_budget_enhancement": "#59A14F",
        "ml_early_stress_risk_budget": "#F28E2B",
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for strategy in strategies:
        frame = strategy_frame(daily, strategy)
        if frame.empty:
            continue
        y = pd.to_numeric(frame["equity"], errors="coerce")
        start = y.dropna().iloc[0]
        ax.plot(
            frame["date"],
            y / start if start else y,
            linewidth=2.0,
            color=colors[strategy],
            label=display_strategy(strategy),
        )
    ax.set_title("Equity Curve: Benchmark vs ML Risk-Budget Strategies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity Growth (Start = 1.0)")
    ax.legend(loc="best", frameon=False)
    format_date_axis(ax)
    return save_png(fig, "fig1_equity_curve.png")


def plot_drawdown_curve(daily: pd.DataFrame) -> Path:
    strategies = [
        "buy_hold",
        "ml_risk_budget_enhancement",
        "ml_early_stress_risk_budget",
    ]
    colors = {
        "buy_hold": "#4C78A8",
        "ml_risk_budget_enhancement": "#59A14F",
        "ml_early_stress_risk_budget": "#F28E2B",
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    for strategy in strategies:
        frame = strategy_frame(daily, strategy)
        if frame.empty:
            continue
        drawdown = pd.to_numeric(frame["drawdown"], errors="coerce")
        ax.plot(
            frame["date"],
            drawdown,
            linewidth=2.0,
            color=colors[strategy],
            label=display_strategy(strategy),
        )
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Drawdown Curve: Benchmark vs ML Risk-Budget Strategies")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", frameon=False)
    format_date_axis(ax)
    return save_png(fig, "fig2_drawdown_curve.png")


def plot_strategy_metrics(metrics: pd.DataFrame) -> Path:
    strategies = [
        "buy_hold",
        "ml_risk_budget_enhancement",
        "ml_early_stress_risk_budget",
    ]
    rows = metrics.loc[metrics["strategy"].isin(strategies)].copy()
    rows["strategy"] = pd.Categorical(rows["strategy"], categories=strategies, ordered=True)
    rows = rows.sort_values("strategy")
    labels = [display_strategy(s) for s in rows["strategy"].astype(str)]
    metric_specs = [
        ("total_return", "Total Return", False),
        ("max_drawdown", "Max Drawdown", True),
        ("sharpe", "Sharpe Ratio", False),
        ("calmar", "Calmar Ratio", False),
    ]
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    for ax, (col, title, percent_axis) in zip(axes.ravel(), metric_specs):
        values = pd.to_numeric(rows[col], errors="coerce").to_numpy()
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        if percent_axis:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.axhline(0, color="#444444", linewidth=0.8)
        for bar, value in zip(bars, values):
            if not np.isfinite(value):
                continue
            text = f"{value:.1%}" if percent_axis else f"{value:.3f}"
            va = "bottom" if value >= 0 else "top"
            offset = 3 if value >= 0 else -3
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
            )
    fig.suptitle("Strategy Metrics Comparison", y=1.02, fontsize=14)
    return save_png(fig, "fig3_strategy_metrics.png")


def plot_position_exposure(daily: pd.DataFrame) -> Path:
    strategy = "ml_risk_budget_enhancement"
    frame = strategy_frame(daily, strategy)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(
        frame["date"],
        pd.to_numeric(frame["final_position"], errors="coerce"),
        linewidth=1.9,
        color="#59A14F",
        label="Risk Budget Position",
    )
    ax.fill_between(
        frame["date"],
        pd.to_numeric(frame["final_position"], errors="coerce"),
        1.0,
        color="#59A14F",
        alpha=0.12,
    )
    ax.set_title("Position Exposure: Risk-Budget Enhancement")
    ax.set_xlabel("Date")
    ax.set_ylabel("Position Weight")
    ax.set_ylim(0.76, 1.02)
    ax.legend(loc="lower left", frameon=False)
    format_date_axis(ax)
    return save_png(fig, "fig4_position_exposure.png")


def plot_model_stability(stability_summary: pd.DataFrame) -> Path:
    labels = stability_summary["model"].tolist()
    x = np.arange(len(labels))
    width = 0.24
    colors = {
        "validation_score": "#4C78A8",
        "test_score": "#59A14F",
        "score_gap": "#E15759",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    metrics = [
        ("validation_score", "Validation Score"),
        ("test_score", "Test Score"),
        ("score_gap", "Score Gap"),
    ]
    for idx, (col, label) in enumerate(metrics):
        values = pd.to_numeric(stability_summary[col], errors="coerce").to_numpy()
        ax.bar(x + (idx - 1) * width, values, width=width, color=colors[col], label=label)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Selection Score")
    ax.set_title("Model Stability Audit: Validation Strength vs Test Generalization")
    ax.legend(loc="upper right", frameon=False)
    mlp_row = stability_summary.loc[stability_summary["model"] == "MLP"]
    if not mlp_row.empty:
        mlp_test = float(mlp_row.iloc[0]["test_score"])
        ax.annotate(
            "MLP validation winner,\nweak test generalization",
            xy=(0, mlp_test),
            xytext=(0.70, 0.72),
            arrowprops={"arrowstyle": "->", "color": "#E15759", "lw": 1.2},
            fontsize=9,
            color="#8C2D2D",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#E15759", "alpha": 0.85},
        )
    return save_png(fig, "fig5_model_stability.png")


def plot_prediction_auc(auc_summary: pd.DataFrame) -> Path:
    labels = auc_summary["model"].tolist()
    x = np.arange(len(labels))
    width = 0.24
    specs = [
        ("test_ordinary_direction_auc", "Ordinary Direction AUC", "#4C78A8"),
        ("test_clean_direction_auc", "Clean-Direction AUC", "#59A14F"),
        ("test_big_up_auc", "Big-Up AUC", "#F28E2B"),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for idx, (col, label, color) in enumerate(specs):
        values = pd.to_numeric(auc_summary[col], errors="coerce").to_numpy()
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=color)
    ax.axhline(0.5, color="#444444", linewidth=0.9, linestyle="--", label="Random baseline")
    ax.set_ylim(0.3, 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Test AUC")
    ax.set_title("Prediction AUC Comparison on the Test Period")
    ax.legend(loc="upper left", frameon=False, ncol=2)
    return save_png(fig, "fig6_prediction_auc.png")


def find_feature_importance_table() -> Path | None:
    candidates = []
    for pattern in ["*feature*importance*.csv", "*importance*feature*.csv"]:
        candidates.extend(CURRENT_DIR.rglob(pattern))
    for path in candidates:
        if "report_figures" in path.parts or "report_tables" in path.parts:
            continue
        return path
    return None


def plot_or_note_feature_importance() -> tuple[Path | None, Path | None]:
    source = find_feature_importance_table()
    if source is None:
        note = FIGURE_DIR / "feature_importance_missing.md"
        note.write_text(
            "\n".join(
                [
                    "# LightGBM 5D Feature Importance Missing",
                    "",
                    "Figure 7 (`fig7_lightgbm_feature_importance.png`) was not generated because the current `outputs/current/` artifacts do not contain a saved LightGBM 5D model object or a saved feature-importance table for the final `ml_risk_budget_enhancement` strategy.",
                    "",
                    "To preserve the no-retraining rule for the final report material pass, this script did not refit LightGBM or recompute feature importance.",
                    "",
                    "Recommended report handling: mention that feature importance was unavailable in the frozen output bundle and keep the model-stability and strategy-attribution figures as the interpretability evidence.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return None, note

    importance = pd.read_csv(source)
    feature_col = next((c for c in importance.columns if c.lower() in {"feature", "feature_name"}), None)
    value_col = next(
        (
            c
            for c in importance.columns
            if c.lower() in {"importance", "gain", "split", "feature_importance"}
        ),
        None,
    )
    if feature_col is None or value_col is None:
        note = FIGURE_DIR / "feature_importance_missing.md"
        note.write_text(
            f"# LightGBM 5D Feature Importance Missing\n\nFound `{source}` but it does not contain recognizable feature and importance columns.\n",
            encoding="utf-8",
        )
        return None, note

    top = importance[[feature_col, value_col]].dropna().copy()
    top[value_col] = pd.to_numeric(top[value_col], errors="coerce")
    top = top.dropna().sort_values(value_col, ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.barh(top[feature_col].astype(str), top[value_col], color="#4C78A8")
    ax.set_title("LightGBM 5D Top 20 Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    return save_png(fig, "fig7_lightgbm_feature_importance.png"), None


def plot_external_overlay(metrics: pd.DataFrame) -> Path:
    strategies = [
        "buy_hold",
        "ml_risk_budget_enhancement",
        "ml_external_soft_lite_risk_budget_index_only",
    ]
    rows = metrics.loc[metrics["strategy"].isin(strategies)].copy()
    rows["strategy"] = pd.Categorical(rows["strategy"], categories=strategies, ordered=True)
    rows = rows.sort_values("strategy")
    labels = [display_strategy(s) for s in rows["strategy"].astype(str)]
    specs = [
        ("total_return", "Total Return", False),
        ("max_drawdown", "Max Drawdown", True),
        ("missed_upside", "Missed Upside", True),
        ("avoided_downside", "Avoided Downside", True),
    ]
    colors = ["#4C78A8", "#59A14F", "#B07AA1"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0))
    for ax, (col, title, percent_axis) in zip(axes.ravel(), specs):
        values = pd.to_numeric(rows[col], errors="coerce").to_numpy()
        ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        if percent_axis:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.axhline(0, color="#444444", linewidth=0.8)
    fig.suptitle("External Market Overlay Experiment", y=1.02, fontsize=14)
    return save_png(fig, "fig8_external_overlay.png")


def parse_external_feature_count() -> str:
    report = CURRENT_DIR / "external_market_feature_report.md"
    if not report.exists():
        return ""
    text = report.read_text(encoding="utf-8")
    match = re.search(r"External feature count in `feature_df`: ([0-9]+)", text)
    return match.group(1) if match else ""


def build_dataset_summary() -> pd.DataFrame:
    feature_columns_path = CURRENT_DIR / "feature_columns.json"
    feature_info = json.loads(feature_columns_path.read_text(encoding="utf-8"))
    validation_predictions = pd.read_csv(CURRENT_DIR / "misc" / "validation_predictions_all_models.csv")
    test_predictions = pd.read_csv(CURRENT_DIR / "misc" / "test_predictions_all_models.csv")
    daily = read_csv("strategy_daily_all.csv")
    label_balance = pd.read_csv(CURRENT_DIR / "diagnostics" / "label_balance_by_horizon.csv")

    rows = [
        {
            "category": "Project",
            "item": "Index universe",
            "value": "880823 micro-cap index",
            "source": "outputs/current/CURRENT_RESULT_README.md",
        },
        {
            "category": "Backtest",
            "item": "Initial capital",
            "value": "RMB 100,000",
            "source": "outputs/current/summary.md",
        },
        {
            "category": "Backtest",
            "item": "Transaction cost rate",
            "value": "0.0010",
            "source": "outputs/current/summary.md",
        },
        {
            "category": "Backtest",
            "item": "Strategy test period",
            "value": f"{daily['trade_date'].min()} to {daily['trade_date'].max()}",
            "source": "outputs/current/strategy_daily_all.csv",
        },
        {
            "category": "Backtest",
            "item": "Unique strategy trading dates",
            "value": int(daily["trade_date"].nunique()),
            "source": "outputs/current/strategy_daily_all.csv",
        },
        {
            "category": "Prediction",
            "item": "Validation prediction date range",
            "value": f"{validation_predictions['date'].min()} to {validation_predictions['date'].max()}",
            "source": "outputs/current/misc/validation_predictions_all_models.csv",
        },
        {
            "category": "Prediction",
            "item": "Test prediction date range",
            "value": f"{test_predictions['date'].min()} to {test_predictions['date'].max()}",
            "source": "outputs/current/misc/test_predictions_all_models.csv",
        },
        {
            "category": "Prediction",
            "item": "Evaluated model families",
            "value": int(validation_predictions["model_name"].nunique()),
            "source": "outputs/current/misc/validation_predictions_all_models.csv",
        },
        {
            "category": "Prediction",
            "item": "Prediction horizons",
            "value": ", ".join(str(int(h)) + "D" for h in sorted(validation_predictions["horizon"].dropna().unique())),
            "source": "outputs/current/misc/validation_predictions_all_models.csv",
        },
        {
            "category": "Features",
            "item": "Total feature count",
            "value": feature_info.get("feature_count", ""),
            "source": "outputs/current/feature_columns.json",
        },
        {
            "category": "Features",
            "item": "Early-stress feature count",
            "value": feature_info.get("early_stress_feature_count", ""),
            "source": "outputs/current/feature_columns.json",
        },
        {
            "category": "Features",
            "item": "External market feature count",
            "value": parse_external_feature_count(),
            "source": "outputs/current/external_market_feature_report.md",
        },
    ]
    for _, row in label_balance.iterrows():
        rows.append(
            {
                "category": "Label balance",
                "item": f"{int(row['horizon'])}D target rates",
                "value": (
                    f"up={row['target_up']:.4f}; trade={row['target_trade']:.4f}; "
                    f"big_up={row['target_big_up']:.4f}; big_down={row['target_big_down']:.4f}; "
                    f"clean_direction={row['target_clean_direction']:.4f}"
                ),
                "source": "outputs/current/diagnostics/label_balance_by_horizon.csv",
            }
        )
    return pd.DataFrame(rows)


def build_model_summary(utility: pd.DataFrame, best_model: dict) -> pd.DataFrame:
    rows = []
    best_validation = best_model.get("best_validation", {})
    matching_test = best_model.get("matching_test", {})
    rows.append(
        {
            "role": "Prediction-layer validation winner",
            "strategy_or_use": "Prediction model selection",
            "model_name": best_validation.get("model_name"),
            "horizon": best_validation.get("horizon"),
            "validation_score": best_validation.get("selection_score"),
            "test_score": matching_test.get("selection_score"),
            "score_gap": (
                best_validation.get("selection_score") - matching_test.get("selection_score")
                if best_validation.get("selection_score") is not None
                and matching_test.get("selection_score") is not None
                else np.nan
            ),
            "test_ordinary_direction_auc": matching_test.get("AUC"),
            "test_clean_direction_auc": matching_test.get("clean_direction_AUC"),
            "test_big_up_auc": matching_test.get("big_up_AUC"),
            "strategy_total_return": np.nan,
            "comment": "Strong validation score but weak test generalization.",
        }
    )
    for _, row in utility.iterrows():
        rows.append(
            {
                "role": "Strategy-layer utility candidate",
                "strategy_or_use": row["strategy_name"],
                "model_name": row["model_name"],
                "horizon": row["horizon"],
                "validation_score": row["prediction_validation_score"],
                "test_score": row["prediction_test_score"],
                "score_gap": row["prediction_score_gap"],
                "test_ordinary_direction_auc": np.nan,
                "test_clean_direction_auc": np.nan,
                "test_big_up_auc": np.nan,
                "strategy_total_return": row["total_return"],
                "comment": row["strategy_utility_comment"],
            }
        )
    return pd.DataFrame(rows)


def build_strategy_results(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    output.insert(1, "strategy_label", output["strategy"].map(display_strategy))
    preferred = [
        "strategy",
        "strategy_label",
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
        "net_timing_contribution",
        "benchmark_clone",
        "outperforms_buy_hold",
        "reduces_max_drawdown",
    ]
    return output[[col for col in preferred if col in output.columns]]


def build_external_experiment(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics.loc[
        (metrics["strategy"].isin(["buy_hold", "ml_risk_budget_enhancement"]))
        | (metrics["strategy"].str.contains("external", na=False))
    ].copy()
    selected.insert(1, "strategy_label", selected["strategy"].map(display_strategy))
    selected["figure8_included"] = selected["strategy"].isin(
        [
            "buy_hold",
            "ml_risk_budget_enhancement",
            "ml_external_soft_lite_risk_budget_index_only",
        ]
    )
    preferred = [
        "strategy",
        "strategy_label",
        "figure8_included",
        "strategy_role",
        "total_return",
        "excess_return_vs_buy_hold",
        "max_drawdown",
        "drawdown_improvement_vs_buy_hold",
        "sharpe",
        "calmar",
        "average_position",
        "days_below_full_exposure",
        "total_turnover",
        "total_transaction_cost",
        "missed_upside",
        "avoided_downside",
        "net_timing_contribution",
        "outperforms_buy_hold",
        "reduces_max_drawdown",
    ]
    return selected[[col for col in preferred if col in selected.columns]]


def write_figure_index(feature_note: Path | None, feature_figure: Path | None) -> Path:
    fig7_filename = (
        feature_figure.name if feature_figure is not None else feature_note.name if feature_note is not None else "missing"
    )
    rows = [
        (
            "Figure 1",
            "fig1_equity_curve.png",
            "Equity growth comparison for buy-and-hold, the risk-budget enhancement, and the early-stress risk-budget strategy.",
            "Strategy Results",
        ),
        (
            "Figure 2",
            "fig2_drawdown_curve.png",
            "Drawdown-path comparison for buy-and-hold, the risk-budget enhancement, and the early-stress risk-budget strategy.",
            "Risk and Drawdown Analysis",
        ),
        (
            "Figure 3",
            "fig3_strategy_metrics.png",
            "Side-by-side comparison of total return, maximum drawdown, Sharpe ratio, and Calmar ratio for the benchmark and two ML risk-budget strategies.",
            "Strategy Results",
        ),
        (
            "Figure 4",
            "fig4_position_exposure.png",
            "Daily position exposure of the final risk-budget enhancement strategy, showing that the strategy remains a high-participation index enhancement rather than a 0/1 timing rule.",
            "Strategy Design",
        ),
        (
            "Figure 5",
            "fig5_model_stability.png",
            "Validation-test stability audit across representative models; the MLP validation winner is highlighted because its test score collapses.",
            "Model Stability Audit",
        ),
        (
            "Figure 6",
            "fig6_prediction_auc.png",
            "Test-period AUC comparison for ordinary direction, clean-direction, and big-up prediction targets across representative models.",
            "Prediction Performance",
        ),
        (
            "Figure 7",
            fig7_filename,
            "LightGBM 5D feature importance. Not generated from the frozen output bundle if no saved importance table or model artifact is available.",
            "Feature Importance / Appendix",
        ),
        (
            "Figure 8",
            "fig8_external_overlay.png",
            "External market overlay experiment comparing return, drawdown, missed upside, and avoided downside for buy-and-hold, the main risk-budget strategy, and the soft-lite index-only overlay.",
            "External Market Data Experiment",
        ),
    ]
    lines = [
        "# Figure Index",
        "",
        "| Figure Number | Filename | Caption | Suggested Report Section |",
        "| --- | --- | --- | --- |",
    ]
    for fig_no, filename, caption, section in rows:
        lines.append(f"| {fig_no} | `{filename}` | {caption} | {section} |")
    path = FIGURE_DIR / "figure_index.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    configure_matplotlib()
    ensure_dirs()

    daily = read_csv("strategy_daily_all.csv")
    final_metrics = read_csv("final_strategy_comparison.csv")
    stability = read_csv("model_stability_audit.csv")
    utility = read_csv("model_strategy_utility_comparison.csv")
    validation_metrics = pd.read_csv(CURRENT_DIR / "diagnostics" / "validation_model_metrics_by_horizon.csv")
    test_metrics = pd.read_csv(CURRENT_DIR / "diagnostics" / "test_model_metrics_by_horizon.csv")
    best_model = json.loads((CURRENT_DIR / "best_model_selection.json").read_text(encoding="utf-8"))

    stability_summary = representative_model_scores(stability)
    auc_summary = representative_prediction_auc(validation_metrics, test_metrics)

    figures = [
        plot_equity_curve(daily),
        plot_drawdown_curve(daily),
        plot_strategy_metrics(final_metrics),
        plot_position_exposure(daily),
        plot_model_stability(stability_summary),
        plot_prediction_auc(auc_summary),
        plot_external_overlay(final_metrics),
    ]
    feature_figure, feature_note = plot_or_note_feature_importance()
    figure_index = write_figure_index(feature_note, feature_figure)

    tables = [
        write_csv(build_dataset_summary(), "Table1_dataset_summary.csv"),
        write_csv(build_model_summary(utility, best_model), "Table2_model_summary.csv"),
        write_csv(auc_summary, "Table3_prediction_results.csv"),
        write_csv(build_strategy_results(final_metrics), "Table4_strategy_results.csv"),
        write_csv(stability_summary, "Table5_model_stability_audit.csv"),
        write_csv(build_external_experiment(final_metrics), "Table6_external_market_experiment.csv"),
    ]

    print("Generated figures:")
    for path in figures:
        print(path)
    if feature_figure is not None:
        print(feature_figure)
    if feature_note is not None:
        print(feature_note)
    print(figure_index)
    print("Generated tables:")
    for path in tables:
        print(path)


if __name__ == "__main__":
    main()
