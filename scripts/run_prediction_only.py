from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DIAGNOSTICS_DIR, METRICS_DIR, OUTPUT_DIR, PLOT_DIR, PREDICTION_DIR

warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import load_data
from src.features import build_daily_features, build_intraday_features, get_feature_columns
from src.metrics import (
    compute_factor_effectiveness,
    compute_factor_effectiveness_tail_targets,
    compute_model_metrics,
    select_best_prediction_model,
)
from src.models import build_model_specs
from src.utils import finite_float, write_json_ready
from src.walk_forward import select_validation_feature_top_k, walk_forward_predictions
from src.config import HORIZONS, MIN_TRAIN_SAMPLES, RETRAIN_EVERY, TEST_END, TEST_START


PREDICTION_EXPORT_COLS = [
    "date",
    "trade_date",
    "target_start_date",
    "target_end_date",
    "horizon",
    "model_name",
    "close",
    "fwd_ret_1d",
    "target_ret",
    "target_up",
    "target_trade",
    "target_big_up",
    "target_big_down",
    "target_clean_direction",
    "pred_ret",
    "pred_up_prob",
    "pred_trade_prob",
    "pred_big_up_prob",
    "pred_big_down_prob",
    "pred_tail_score",
    "pred_clean_direction_prob",
    "pred_clean_direction",
    "pred_up",
    "pred_trade",
    "pred_big_up",
    "pred_big_down",
    "chunk_start",
    "train_samples",
    "train_latest_target_end_date",
    "feature_top_k",
    "selected_feature_count",
    "selected_feature_count_up",
    "selected_feature_count_trade",
    "selected_feature_count_big_up",
    "selected_feature_count_big_down",
    "selected_feature_count_clean_direction",
    "clean_direction_train_samples",
]


def ensure_output_dirs() -> None:
    for path in [OUTPUT_DIR, PREDICTION_DIR, METRICS_DIR, DIAGNOSTICS_DIR, PLOT_DIR, ROOT / ".mplconfig", ROOT / ".cache"]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(write_json_ready(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def max_metric(metrics: pd.DataFrame, col: str) -> float | None:
    if metrics.empty or col not in metrics.columns:
        return None
    value = metrics[col].replace([np.inf, -np.inf], np.nan).max()
    return finite_float(value)


def fmt(value: Any, digits: int = 4) -> str:
    val = finite_float(value)
    if val is None:
        return "NA"
    return f"{val:.{digits}f}"


def best_row(metrics: pd.DataFrame, col: str) -> dict[str, Any]:
    if metrics.empty or col not in metrics.columns:
        return {}
    ranked = metrics.replace([np.inf, -np.inf], np.nan).dropna(subset=[col]).sort_values(col, ascending=False)
    return ranked.iloc[0].to_dict() if not ranked.empty else {}


def plot_model_metric_comparison(validation_metrics: pd.DataFrame, test_metrics: pd.DataFrame) -> None:
    rows = []
    for period, metrics in [("validation", validation_metrics), ("test", test_metrics)]:
        if metrics.empty:
            continue
        grouped = metrics.groupby("model_name", as_index=False)[
            ["AUC", "clean_direction_AUC", "trade_AUC", "big_up_AUC", "big_down_AUC", "selection_score"]
        ].max()
        grouped["period"] = period
        rows.append(grouped)
    if not rows:
        return
    plot_df = pd.concat(rows, ignore_index=True)
    plot_df["label"] = plot_df["model_name"] + " " + plot_df["period"]
    plot_df = plot_df.sort_values("selection_score", ascending=False).head(18)

    plt.figure(figsize=(13, 7))
    x = np.arange(len(plot_df))
    width = 0.18
    for offset, col in [(-1.5, "AUC"), (-0.5, "clean_direction_AUC"), (0.5, "trade_AUC"), (1.5, "selection_score")]:
        plt.bar(x + offset * width, plot_df[col].fillna(0.5), width=width, label=col)
    plt.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    plt.xticks(x, plot_df["label"], rotation=45, ha="right")
    plt.ylabel("Metric")
    plt.title("Prediction Model Metric Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_metric_comparison.png", dpi=160)
    plt.close()


def plot_best_prediction_diagnostics(test_preds: pd.DataFrame, best_model: dict[str, Any]) -> None:
    best = best_model["best_validation"]
    model_name = best["model_name"]
    horizon = int(best["horizon"])
    sample = test_preds.loc[(test_preds["model_name"] == model_name) & (test_preds["horizon"] == horizon)].copy()
    if sample.empty:
        fallback = best_row(compute_model_metrics(test_preds), "selection_score")
        if not fallback:
            return
        model_name = fallback["model_name"]
        horizon = int(fallback["horizon"])
        sample = test_preds.loc[(test_preds["model_name"] == model_name) & (test_preds["horizon"] == horizon)].copy()
    sample = sample.sort_values("date")

    plt.figure(figsize=(13, 6))
    plt.plot(sample["date"], sample["pred_up_prob"], label="up probability", linewidth=1.6)
    plt.plot(sample["date"], sample["pred_clean_direction_prob"], label="clean-direction probability", linewidth=1.6)
    clean_points = sample.dropna(subset=["target_clean_direction"])
    if not clean_points.empty:
        plt.scatter(
            clean_points["date"],
            clean_points["target_clean_direction"],
            label="clean-direction target",
            alpha=0.45,
            s=14,
        )
    plt.ylim(-0.05, 1.05)
    plt.title(f"Predicted Probability Over Time: {model_name}, {horizon}D")
    plt.xlabel("Date")
    plt.ylabel("Probability / Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "predicted_probability_over_time_best_model.png", dpi=160)
    plt.close()

    scatter = sample.dropna(subset=["target_ret", "pred_ret"])
    if not scatter.empty:
        plt.figure(figsize=(8, 7))
        plt.scatter(scatter["pred_ret"], scatter["target_ret"], alpha=0.65, s=22)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title(f"Predicted vs Actual Forward Return: {model_name}, {horizon}D")
        plt.xlabel("Predicted return")
        plt.ylabel("Actual forward return")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "prediction_scatter_best_model.png", dpi=160)
        plt.close()


def write_prediction_summary(
    feature_count: int,
    selected_top_k_label: str,
    model_families: list[str],
    best_model: dict[str, Any],
    test_metrics: pd.DataFrame,
) -> None:
    best_validation = best_model["best_validation"]
    best_test_auc = best_row(test_metrics, "AUC")
    best_test_clean = best_row(test_metrics, "clean_direction_AUC")
    best_test_trade = best_row(test_metrics, "trade_AUC")
    best_test_big_up = best_row(test_metrics, "big_up_AUC")
    best_test_big_down = best_row(test_metrics, "big_down_AUC")
    best_test_corr = best_row(test_metrics, "return_correlation")
    clean_reached = (max_metric(test_metrics, "clean_direction_AUC") or 0.0) >= 0.60

    summary = f"""# Prediction Summary

## Run Scope
- Feature count: {feature_count}
- Selected feature_top_k: {selected_top_k_label}
- Model families used: {", ".join(model_families)}
- Horizons: {", ".join(str(h) for h in HORIZONS)}

## Best Validation Selection
- Best validation model: {best_validation.get("model_name")}
- Best validation horizon: {best_validation.get("horizon")}
- Best validation ordinary AUC: {fmt(best_validation.get("AUC"))}
- Best validation clean_direction_AUC: {fmt(best_validation.get("clean_direction_AUC"))}
- Validation selection score: {fmt(best_validation.get("selection_score"))}

## Test Prediction Metrics
- Best test ordinary AUC: {fmt(best_test_auc.get("AUC"))} ({best_test_auc.get("model_name", "NA")}, {best_test_auc.get("horizon", "NA")}D)
- Best test clean_direction_AUC: {fmt(best_test_clean.get("clean_direction_AUC"))} ({best_test_clean.get("model_name", "NA")}, {best_test_clean.get("horizon", "NA")}D)
- Best test trade_AUC: {fmt(best_test_trade.get("trade_AUC"))} ({best_test_trade.get("model_name", "NA")}, {best_test_trade.get("horizon", "NA")}D)
- Best test big_up_AUC: {fmt(best_test_big_up.get("big_up_AUC"))} ({best_test_big_up.get("model_name", "NA")}, {best_test_big_up.get("horizon", "NA")}D)
- Best test big_down_AUC: {fmt(best_test_big_down.get("big_down_AUC"))} ({best_test_big_down.get("model_name", "NA")}, {best_test_big_down.get("horizon", "NA")}D)
- Best test return_correlation: {fmt(best_test_corr.get("return_correlation"))} ({best_test_corr.get("model_name", "NA")}, {best_test_corr.get("horizon", "NA")}D)
- Clean-direction AUC >= 0.60: {"Yes" if clean_reached else "No"}

## Leakage-Control Checklist
- No random train/test split is used.
- Validation period and test period are fixed by configuration.
- Feature_top_k is selected using validation predictions only.
- Model preference is selected using validation prediction metrics only.
- Each walk-forward chunk trains only on rows with target_end_date < chunk_start.
- Feature selection is refit inside each walk-forward chunk using only that chunk's training rows.
- Quantile clipping and median imputation are fitted only on the relevant task's training rows.
- Historical target quantiles only use returns whose target_end_date is before the current date.
- Test-set information is not used for feature selection, model selection, preprocessing, or threshold construction.

## Interpretation
- Raw up/down AUC may remain low because near-zero returns are noisy.
- clean_direction_AUC is the denoised directional metric because middle noisy samples are ignored.
- trade_AUC and tail-target AUC are more relevant for later strategy design than raw up/down alone.
"""
    (DIAGNOSTICS_DIR / "prediction_summary.md").write_text(summary, encoding="utf-8")


def run_prediction_pipeline(print_summary: bool = True) -> dict[str, Any]:
    ensure_output_dirs()

    print("Loading data...")
    daily, m5 = load_data()
    print(f"Daily rows: {len(daily)}, 5-minute rows: {len(m5)}")

    print("Building leakage-safe features and targets...")
    intraday = build_intraday_features(m5)
    feature_df = build_daily_features(daily, intraday)
    feature_cols = get_feature_columns(feature_df)
    model_specs, skipped_models = build_model_specs()
    model_families = [spec.name for spec in model_specs]

    write_json(
        DIAGNOSTICS_DIR / "feature_columns.json",
        {
            "feature_count": len(feature_cols),
            "features": feature_cols,
            "skipped_optional_models": skipped_models,
        },
    )
    compute_factor_effectiveness(feature_df, feature_cols).to_csv(DIAGNOSTICS_DIR / "factor_effectiveness.csv", index=False)
    compute_factor_effectiveness_tail_targets(feature_df, feature_cols).to_csv(
        DIAGNOSTICS_DIR / "factor_effectiveness_tail_targets.csv",
        index=False,
    )

    print(f"Feature count: {len(feature_cols)}")
    print(f"Model families: {model_families}")
    if skipped_models:
        print(f"Skipped optional models: {skipped_models}")

    print("Selecting validation-only feature_top_k and running validation walk-forward...")
    feature_top_k_selection = select_validation_feature_top_k(feature_df, feature_cols, model_specs)
    selected_feature_top_k = feature_top_k_selection["selected_top_k"]
    selected_top_k_label = feature_top_k_selection["selected_top_k_label"]
    validation_preds = feature_top_k_selection["predictions"]
    validation_metrics = compute_model_metrics(validation_preds)

    feature_top_k_export = {k: v for k, v in feature_top_k_selection.items() if k != "predictions"}
    write_json(METRICS_DIR / "feature_top_k_selection.json", feature_top_k_export)
    validation_preds[PREDICTION_EXPORT_COLS].to_csv(PREDICTION_DIR / "validation_predictions_all_models.csv", index=False)
    validation_metrics.to_csv(METRICS_DIR / "validation_model_metrics_by_horizon.csv", index=False)

    print("Running test walk-forward predictions...")
    test_preds = walk_forward_predictions(
        feature_df,
        feature_cols,
        model_specs,
        TEST_START,
        TEST_END,
        RETRAIN_EVERY,
        min_train_samples=MIN_TRAIN_SAMPLES,
        label="test",
        feature_top_k=selected_feature_top_k,
    )
    test_metrics = compute_model_metrics(test_preds)
    test_preds[PREDICTION_EXPORT_COLS].to_csv(PREDICTION_DIR / "test_predictions_all_models.csv", index=False)
    test_metrics.to_csv(METRICS_DIR / "test_model_metrics_by_horizon.csv", index=False)

    best_model = select_best_prediction_model(validation_metrics, test_metrics)
    best_model["feature_top_k_selection"] = feature_top_k_export
    best_model["model_families_used"] = model_families
    best_model["skipped_optional_models"] = skipped_models
    write_json(METRICS_DIR / "best_prediction_model.json", best_model)

    plot_model_metric_comparison(validation_metrics, test_metrics)
    plot_best_prediction_diagnostics(test_preds, best_model)
    write_prediction_summary(len(feature_cols), selected_top_k_label, model_families, best_model, test_metrics)

    best_validation = best_model["best_validation"]
    clean_reached = (max_metric(test_metrics, "clean_direction_AUC") or 0.0) >= 0.60

    if print_summary:
        print("\n===== PREDICTION-ONLY RESULTS =====")
        print(f"Feature count: {len(feature_cols)}")
        print(f"Selected feature top_k: {selected_top_k_label}")
        print(f"Best validation model: {best_validation.get('model_name')}")
        print(f"Best validation horizon: {best_validation.get('horizon')}")
        print(f"Best validation AUC: {fmt(best_validation.get('AUC'))}")
        print(f"Best validation clean_direction_AUC: {fmt(best_validation.get('clean_direction_AUC'))}")
        print(f"Best test ordinary AUC: {fmt(max_metric(test_metrics, 'AUC'))}")
        print(f"Best test clean_direction_AUC: {fmt(max_metric(test_metrics, 'clean_direction_AUC'))}")
        print(f"Best test trade_AUC: {fmt(max_metric(test_metrics, 'trade_AUC'))}")
        print(f"Best test big_up_AUC: {fmt(max_metric(test_metrics, 'big_up_AUC'))}")
        print(f"Best test big_down_AUC: {fmt(max_metric(test_metrics, 'big_down_AUC'))}")
        print(f"Best test return correlation: {fmt(max_metric(test_metrics, 'return_correlation'))}")
        print(f"Clean-direction AUC >= 0.60: {'Yes' if clean_reached else 'No'}")
        print(f"Outputs saved to: {OUTPUT_DIR}")
        print("==================================")

    return {
        "feature_df": feature_df,
        "feature_cols": feature_cols,
        "validation_predictions": validation_preds,
        "test_predictions": test_preds,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "best_prediction_model": best_model,
        "feature_top_k_selection": feature_top_k_export,
        "model_families": model_families,
        "skipped_models": skipped_models,
    }


def main() -> None:
    run_prediction_pipeline(print_summary=True)


if __name__ == "__main__":
    main()
