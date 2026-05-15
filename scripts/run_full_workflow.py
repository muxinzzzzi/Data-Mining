from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.run_prediction_only import fmt, run_prediction_pipeline
from src.config import DIAGNOSTICS_DIR, METRICS_DIR, OUTPUT_DIR, PLOT_DIR, PREDICTION_DIR
from src.data_loader import load_data
from src.features import build_daily_features, build_intraday_features
from src.strategy.attribution import build_main_strategy_attribution, write_attribution_summary
from src.strategy.backtest import build_all_strategy_daily
from src.strategy.metrics import compute_strategy_metrics, select_strategy_rows
from src.strategy.plots import plot_strategy_outputs
from src.strategy.reports import write_strategy_reports
from src.strategy.tuning import tune_all_strategy_families
from src.utils import write_json_ready


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(write_json_ready(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def _export_tuning_info(tuned: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family, info in tuned.items():
        out[family] = {
            "family": info["family"],
            "selected": info["selected"],
            "selected_params": info["selected_params"].to_dict(),
            "candidate_count": info["candidate_count"],
            "top_candidates": info["top_candidates"],
            "source_selection_rule": info["source_selection_rule"],
        }
    return out


def _strategy_value(row: pd.Series | None, col: str) -> Any:
    return None if row is None else row.get(col)


def _parse_prediction_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["date", "trade_date", "target_start_date", "target_end_date", "chunk_start", "train_latest_target_end_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col])
    return out


def _prediction_outputs_available() -> bool:
    required = [
        PREDICTION_DIR / "validation_predictions_all_models.csv",
        PREDICTION_DIR / "test_predictions_all_models.csv",
        METRICS_DIR / "validation_model_metrics_by_horizon.csv",
        METRICS_DIR / "test_model_metrics_by_horizon.csv",
        DIAGNOSTICS_DIR / "feature_columns.json",
    ]
    return all(path.exists() for path in required)


def load_existing_prediction_artifacts() -> dict[str, Any]:
    print("Loading existing prediction artifacts for strategy workflow...")
    daily, m5 = load_data()
    feature_df = build_daily_features(daily, build_intraday_features(m5))
    validation_predictions = _parse_prediction_dates(pd.read_csv(PREDICTION_DIR / "validation_predictions_all_models.csv"))
    test_predictions = _parse_prediction_dates(pd.read_csv(PREDICTION_DIR / "test_predictions_all_models.csv"))
    validation_metrics = pd.read_csv(METRICS_DIR / "validation_model_metrics_by_horizon.csv")
    test_metrics = pd.read_csv(METRICS_DIR / "test_model_metrics_by_horizon.csv")
    return {
        "feature_df": feature_df,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }


def get_prediction_artifacts() -> dict[str, Any]:
    force_rebuild = os.environ.get("FORCE_REBUILD_PREDICTIONS", "0") == "1"
    if _prediction_outputs_available() and not force_rebuild:
        return load_existing_prediction_artifacts()
    return run_prediction_pipeline(print_summary=True)


def main() -> None:
    artifacts = get_prediction_artifacts()
    feature_df = artifacts["feature_df"]
    validation_predictions = artifacts["validation_predictions"]
    test_predictions = artifacts["test_predictions"]
    validation_metrics = artifacts["validation_metrics"]
    test_metrics = artifacts["test_metrics"]

    print("\nTuning strategy parameters on validation period only...")
    tuned = tune_all_strategy_families(feature_df, validation_predictions, validation_metrics)
    selected_params = {family: info["selected_params"] for family, info in tuned.items()}
    tuning_export = _export_tuning_info(tuned)
    write_json(OUTPUT_DIR / "strategy_tuned_params.json", tuning_export)

    print("Backtesting validation-selected strategies on test period...")
    strategy_daily = build_all_strategy_daily(feature_df, test_predictions, selected_params)
    strategy_metrics = compute_strategy_metrics(strategy_daily)
    strategy_daily.to_csv(OUTPUT_DIR / "strategy_daily_all.csv", index=False)
    strategy_metrics.to_csv(OUTPUT_DIR / "strategy_metrics_all.csv", index=False)

    main_strategy = "ml_big_up_index_enhancement"
    attribution = build_main_strategy_attribution(strategy_daily, main_strategy)
    attribution.to_csv(OUTPUT_DIR / "main_strategy_attribution.csv", index=False)
    write_attribution_summary(attribution, OUTPUT_DIR / "main_strategy_attribution_summary.md")

    plot_strategy_outputs(strategy_daily, strategy_metrics, main_strategy)
    write_strategy_reports(
        strategy_metrics,
        test_metrics,
        tuning_export,
        OUTPUT_DIR / "summary.md",
        OUTPUT_DIR / "final_report_material.md",
    )

    rows = select_strategy_rows(strategy_metrics)
    buy_hold = rows["buy_hold"]
    best_no_lev = rows["best_no_leverage"]
    best_real = rows["best_real_no_leverage"]
    best_plus = rows["best_enhanced_exposure"]
    no_leverage = strategy_metrics.loc[
        strategy_metrics["strategy"].isin(
            [
                "ml_big_up_index_enhancement",
                "ml_conservative_full_participation_enhancement",
                "ml_ultra_conservative_full_participation_enhancement",
                "ml_direct_signal_timing",
            ]
        )
        & (~strategy_metrics["benchmark_clone"].astype(bool))
    ]
    any_no_lev_outperform = bool((no_leverage["excess_return_vs_buy_hold"] > 0).any())

    print("\n===== ENHANCED INDEX STRATEGY RESULTS =====")
    print(f"Best no-leverage strategy: {_strategy_value(best_no_lev, 'strategy')}")
    print(f"Whether it is benchmark clone: {'Yes' if bool(_strategy_value(best_no_lev, 'benchmark_clone')) else 'No'}")
    print(f"Best real no-leverage ML strategy: {_strategy_value(best_real, 'strategy')}")
    print(f"Buy-and-hold total return: {fmt(_strategy_value(buy_hold, 'total_return'))}")
    print(f"Buy-and-hold max drawdown: {fmt(_strategy_value(buy_hold, 'max_drawdown'))}")
    print(f"ML strategy total return: {fmt(_strategy_value(best_real, 'total_return'))}")
    print(f"ML strategy excess return: {fmt(_strategy_value(best_real, 'excess_return_vs_buy_hold'))}")
    print(f"ML strategy max drawdown: {fmt(_strategy_value(best_real, 'max_drawdown'))}")
    print(f"Best enhanced-exposure strategy: {_strategy_value(best_plus, 'strategy')}")
    print(f"Best enhanced-exposure excess return: {fmt(_strategy_value(best_plus, 'excess_return_vs_buy_hold'))}")
    print(f"Whether any no-leverage strategy outperforms buy-and-hold: {'Yes' if any_no_lev_outperform else 'No'}")
    print(f"Strategy outputs saved to: {OUTPUT_DIR}")
    print(f"Strategy plots saved to: {PLOT_DIR}")
    print("==========================================")


if __name__ == "__main__":
    main()
