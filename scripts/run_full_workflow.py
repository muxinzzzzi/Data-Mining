from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DIAGNOSTICS_DIR, METRICS_DIR, OUTPUT_DIR, PLOT_DIR, PREDICTION_DIR
from src.data_loader import load_data
from src.features import EARLY_STRESS_FEATURE_CATEGORIES, build_daily_features, build_intraday_features, early_stress_feature_names
from src.model_stability_audit import REQUIRED_AUDIT_MODEL_NAMES, write_model_stability_reports
from src.signal_stability import (
    DEFAULT_SIGNAL_COLS,
    build_stability_aware_upside_score,
    compute_signal_quantile_stability,
    markdown_signal_stability_report,
    select_stability_aware_upside_signals,
    select_tradeable_upside_model,
)
from src.strategy.attribution import build_main_strategy_attribution, write_attribution_summary
from src.strategy.backtest import build_all_strategy_daily
from src.strategy.drawdown_event_diagnostics import (
    print_drawdown_event_diagnostics,
    run_drawdown_event_diagnostics,
)
from src.strategy.external_reports import (
    append_external_summary_section,
    write_external_market_feature_report,
    write_final_strategy_comparison,
)
from src.strategy.metrics import compute_strategy_metrics, select_strategy_rows
from src.strategy.plots import (
    plot_early_stress_risk_budget_outputs,
    plot_risk_budget_outputs,
    plot_stability_aware_high_participation_outputs,
    plot_signal_quantile_stability,
    plot_strategy_outputs,
    plot_upside_participation_outputs,
)
from src.strategy.reports import write_strategy_reports
from src.strategy.tuning import tune_all_strategy_families
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
    build_external_market_ablation_params,
    tune_external_soft_confirmation_params,
    tune_stability_aware_high_participation_params,
    tune_early_stress_risk_budget_params,
    tune_risk_budget_enhancement_params,
    tune_upside_participation_enhancement_params,
    write_early_stress_risk_budget_attribution_summary,
    write_risk_budget_attribution_summary,
    write_stability_aware_high_participation_attribution_summary,
    write_upside_participation_attribution_summary,
)
from src.utils import finite_float, write_json_ready


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(write_json_ready(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    val = finite_float(value)
    return "NA" if val is None else f"{val:.{digits}f}"


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


def _export_upside_tuning_info(tuned: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in tuned.items() if k != "selected_params"}
    out["selected_params"] = tuned["selected_params"].to_dict()
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
    if not all(path.exists() for path in required):
        return False
    try:
        feature_path = DIAGNOSTICS_DIR / "feature_columns.json"
        feature_info = json.loads((DIAGNOSTICS_DIR / "feature_columns.json").read_text(encoding="utf-8"))
        existing = set(feature_info.get("features", []))
        required_early = set(early_stress_feature_names())
        if not required_early.issubset(existing):
            print("Existing prediction artifacts do not include early-stress features; rebuilding predictions.")
            return False
        feature_mtime = feature_path.stat().st_mtime
        stale = [
            path
            for path in required
            if path != feature_path and path.stat().st_mtime < feature_mtime
        ]
        if stale:
            print("Existing prediction artifacts are older than feature_columns.json; rebuilding predictions.")
            return False
        validation_metrics = pd.read_csv(METRICS_DIR / "validation_model_metrics_by_horizon.csv")
        existing_models = set(validation_metrics["model_name"].astype(str))
        missing_models = REQUIRED_AUDIT_MODEL_NAMES - existing_models
        if missing_models:
            print(f"Existing prediction artifacts do not include linear audit baselines {sorted(missing_models)}; rebuilding predictions.")
            return False
        required_metric_cols = {"logloss", "trade_logloss", "big_up_logloss", "big_down_logloss", "clean_direction_logloss"}
        missing_metric_cols = required_metric_cols - set(validation_metrics.columns)
        if missing_metric_cols:
            print(f"Existing prediction metrics do not include logloss columns {sorted(missing_metric_cols)}; rebuilding predictions.")
            return False
    except Exception:
        return False
    return True


def _copy_current_result_files(filenames: list[str]) -> None:
    current = OUTPUT_DIR / "current"
    current.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = OUTPUT_DIR / filename
        if source.exists():
            shutil.copy2(source, current / filename)


def load_existing_prediction_artifacts() -> dict[str, Any]:
    print("Loading existing prediction artifacts for strategy workflow...")
    daily, m5 = load_data()
    feature_df = build_daily_features(daily, build_intraday_features(m5))
    validation_predictions = _parse_prediction_dates(pd.read_csv(PREDICTION_DIR / "validation_predictions_all_models.csv"))
    test_predictions = _parse_prediction_dates(pd.read_csv(PREDICTION_DIR / "test_predictions_all_models.csv"))
    validation_metrics = pd.read_csv(METRICS_DIR / "validation_model_metrics_by_horizon.csv")
    test_metrics = pd.read_csv(METRICS_DIR / "test_model_metrics_by_horizon.csv")
    best_prediction_model_path = METRICS_DIR / "best_prediction_model.json"
    best_prediction_model = (
        json.loads(best_prediction_model_path.read_text(encoding="utf-8"))
        if best_prediction_model_path.exists()
        else {}
    )
    return {
        "feature_df": feature_df,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "best_prediction_model": best_prediction_model,
    }


def get_prediction_artifacts() -> dict[str, Any]:
    force_rebuild = os.environ.get("FORCE_REBUILD_PREDICTIONS", "0") == "1"
    if _prediction_outputs_available() and not force_rebuild:
        return load_existing_prediction_artifacts()
    from scripts.run_prediction_only import run_prediction_pipeline

    return run_prediction_pipeline(print_summary=True)


def main() -> None:
    artifacts = get_prediction_artifacts()
    feature_df = artifacts["feature_df"]
    validation_predictions = artifacts["validation_predictions"]
    test_predictions = artifacts["test_predictions"]
    validation_metrics = artifacts["validation_metrics"]
    test_metrics = artifacts["test_metrics"]
    best_prediction_model = artifacts.get("best_prediction_model", {})

    print("\nComputing signal quantile stability diagnostics...")
    validation_for_stability = validation_predictions.copy()
    validation_for_stability["period"] = "validation"
    test_for_stability = test_predictions.copy()
    test_for_stability["period"] = "test"
    signal_bucket, signal_summary = compute_signal_quantile_stability(
        pd.concat([validation_for_stability, test_for_stability], ignore_index=True),
        DEFAULT_SIGNAL_COLS,
    )
    signal_bucket.to_csv(OUTPUT_DIR / "signal_quantile_stability.csv", index=False)
    signal_summary.to_csv(OUTPUT_DIR / "signal_quantile_stability_summary.csv", index=False)

    print("Selecting tradeable-upside signal model using validation diagnostics only...")
    upside_signal_selection = select_tradeable_upside_model(validation_metrics, signal_summary)
    write_json(OUTPUT_DIR / "upside_signal_model_selection.json", upside_signal_selection)
    (OUTPUT_DIR / "signal_quantile_stability_summary.md").write_text(
        markdown_signal_stability_report(signal_summary, upside_signal_selection),
        encoding="utf-8",
    )
    plot_signal_quantile_stability(signal_bucket, upside_signal_selection)

    print("Selecting stability-aware upside signal components using validation diagnostics only...")
    stability_aware_signal_selection = select_stability_aware_upside_signals(
        signal_summary,
        validation_metrics,
        max_components=3,
    )
    write_json(OUTPUT_DIR / "stability_aware_signal_selection.json", stability_aware_signal_selection)
    selected_components = stability_aware_signal_selection["selected_components"]
    validation_stability_score = build_stability_aware_upside_score(validation_predictions, selected_components)
    validation_stability_score["period"] = "validation"
    test_stability_score = build_stability_aware_upside_score(test_predictions, selected_components)
    test_stability_score["period"] = "test"
    stability_score_all = pd.concat([validation_stability_score, test_stability_score], ignore_index=True)
    stability_score_all.to_csv(OUTPUT_DIR / "stability_aware_upside_score.csv", index=False)

    print("\nTuning strategy parameters on validation period only...")
    tuned = tune_all_strategy_families(feature_df, validation_predictions, validation_metrics)
    selected_params = {family: info["selected_params"] for family, info in tuned.items()}
    tuning_export = _export_tuning_info(tuned)
    write_json(OUTPUT_DIR / "strategy_tuned_params.json", tuning_export)

    print("Tuning upside participation enhancement parameters on validation period only...")
    upside_tuned = tune_upside_participation_enhancement_params(
        feature_df,
        validation_predictions,
        upside_signal_selection,
    )
    write_json(OUTPUT_DIR / "upside_participation_tuned_params.json", _export_upside_tuning_info(upside_tuned))
    selected_params[UPSIDE_STRATEGY_NAME] = upside_tuned["selected_params"]

    print("Tuning stability-aware high-participation parameters on validation period only...")
    stability_tuned = tune_stability_aware_high_participation_params(
        feature_df,
        validation_predictions,
        selected_components,
    )
    write_json(
        OUTPUT_DIR / "stability_aware_high_participation_tuned_params.json",
        _export_upside_tuning_info(stability_tuned),
    )
    selected_params[STABILITY_AWARE_STRATEGY_NAME] = stability_tuned["selected_params"]

    print("Tuning risk-budget drawdown-control parameters on validation period only...")
    risk_budget_tuned = tune_risk_budget_enhancement_params(
        feature_df,
        validation_predictions,
    )
    write_json(OUTPUT_DIR / "risk_budget_tuned_params.json", _export_upside_tuning_info(risk_budget_tuned))
    selected_params[RISK_BUDGET_STRATEGY_NAME] = risk_budget_tuned["selected_params"]

    print("Tuning early-stress risk-budget parameters on validation period only...")
    early_stress_tuned = tune_early_stress_risk_budget_params(
        feature_df,
        validation_predictions,
    )
    write_json(OUTPUT_DIR / "early_stress_risk_budget_tuned_params.json", _export_upside_tuning_info(early_stress_tuned))
    selected_params[EARLY_STRESS_STRATEGY_NAME] = early_stress_tuned["selected_params"]

    print("Building external market data ablation strategies from validation-selected risk params...")
    external_ablation_params = build_external_market_ablation_params(
        risk_budget_tuned["selected_params"],
        early_stress_tuned["selected_params"],
    )
    selected_params.update(external_ablation_params)
    write_json(
        OUTPUT_DIR / "external_market_ablation_params.json",
        {name: params.to_dict() for name, params in external_ablation_params.items()},
    )

    print("Tuning soft external confirmation strategies on validation period only...")
    external_soft_tuned = tune_external_soft_confirmation_params(
        feature_df,
        validation_predictions,
        risk_budget_tuned["selected_params"],
        early_stress_tuned["selected_params"],
    )
    external_soft_params = {family: info["selected_params"] for family, info in external_soft_tuned.items()}
    selected_params.update(external_soft_params)
    write_json(
        OUTPUT_DIR / "external_soft_confirmation_tuned_params.json",
        {
            family: {
                **{k: v for k, v in info.items() if k != "selected_params"},
                "selected_params": info["selected_params"].to_dict(),
            }
            for family, info in external_soft_tuned.items()
        },
    )

    print("Backtesting validation-selected strategies on test period...")
    test_predictions_with_stability = test_predictions.merge(
        test_stability_score.drop(columns=["period"], errors="ignore"),
        on="date",
        how="left",
    )
    strategy_daily = build_all_strategy_daily(feature_df, test_predictions_with_stability, selected_params)
    strategy_metrics = compute_strategy_metrics(strategy_daily)
    strategy_daily.to_csv(OUTPUT_DIR / "strategy_daily_all.csv", index=False)
    strategy_metrics.to_csv(OUTPUT_DIR / "strategy_metrics_all.csv", index=False)

    main_strategy = "ml_big_up_index_enhancement"
    attribution = build_main_strategy_attribution(strategy_daily, main_strategy)
    attribution.to_csv(OUTPUT_DIR / "main_strategy_attribution.csv", index=False)
    write_attribution_summary(attribution, OUTPUT_DIR / "main_strategy_attribution_summary.md")
    write_upside_participation_attribution_summary(
        strategy_daily,
        upside_signal_selection["selected"]["signal_name"],
        OUTPUT_DIR / "upside_participation_attribution_summary.md",
    )
    write_stability_aware_high_participation_attribution_summary(
        strategy_daily,
        selected_components,
        OUTPUT_DIR / "stability_aware_high_participation_attribution_summary.md",
    )
    write_risk_budget_attribution_summary(
        strategy_daily,
        OUTPUT_DIR / "risk_budget_attribution_summary.md",
    )
    write_early_stress_risk_budget_attribution_summary(
        strategy_daily,
        OUTPUT_DIR / "early_stress_risk_budget_attribution_summary.md",
    )

    plot_strategy_outputs(strategy_daily, strategy_metrics, main_strategy)
    plot_upside_participation_outputs(
        strategy_daily,
        strategy_metrics,
        upside_signal_selection["selected"]["signal_name"],
    )
    plot_stability_aware_high_participation_outputs(
        strategy_daily,
        strategy_metrics,
        selected_components,
    )
    plot_risk_budget_outputs(strategy_daily, strategy_metrics)
    plot_early_stress_risk_budget_outputs(strategy_daily, strategy_metrics)
    drawdown_event_report = run_drawdown_event_diagnostics(strategy_daily, OUTPUT_DIR, PLOT_DIR)
    write_strategy_reports(
        strategy_metrics,
        test_metrics,
        tuning_export,
        OUTPUT_DIR / "summary.md",
        OUTPUT_DIR / "final_report_material.md",
        tradeable_up_context={
            "selection": upside_signal_selection,
            "stability_summary": signal_summary,
            "upside_tuned": _export_upside_tuning_info(upside_tuned),
        },
        stability_aware_context={
            "selection": stability_aware_signal_selection,
            "selected_components": selected_components,
            "stability_tuned": _export_upside_tuning_info(stability_tuned),
        },
        risk_budget_context={
            "risk_budget_tuned": _export_upside_tuning_info(risk_budget_tuned),
        },
        early_stress_context={
            "early_stress_tuned": _export_upside_tuning_info(early_stress_tuned),
            "feature_categories": EARLY_STRESS_FEATURE_CATEGORIES,
            "drawdown_event_report": drawdown_event_report,
        },
        prediction_validation_metrics=validation_metrics,
    )
    final_strategy_comparison = write_final_strategy_comparison(strategy_metrics, OUTPUT_DIR)
    write_external_market_feature_report(feature_df, final_strategy_comparison, ROOT / "raw", OUTPUT_DIR)
    append_external_summary_section(OUTPUT_DIR / "summary.md", final_strategy_comparison)
    write_model_stability_reports(
        feature_df=feature_df,
        test_predictions=test_predictions_with_stability,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        strategy_metrics=strategy_metrics,
        best_prediction_model=best_prediction_model,
        strategy_params=selected_params,
        upside_params=upside_tuned["selected_params"],
        risk_budget_params=risk_budget_tuned["selected_params"],
        early_stress_params=early_stress_tuned["selected_params"],
        output_dir=OUTPUT_DIR,
        summary_path=OUTPUT_DIR / "summary.md",
    )
    _copy_current_result_files(
        [
            "summary.md",
            "final_strategy_comparison.csv",
            "final_strategy_comparison.md",
            "external_market_feature_report.md",
            "model_stability_audit.csv",
            "model_stability_audit.md",
            "model_strategy_utility_comparison.csv",
            "model_strategy_utility_comparison.md",
        ]
    )

    try:
        from scripts.organize_outputs import main as organize_outputs_main

        organize_outputs_main()
    except Exception as exc:
        print(f"Output organization skipped: {exc}")

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
        & (~strategy_metrics["benchmark_clone"].astype(bool))
    ]
    any_no_lev_outperform = bool((no_leverage["excess_return_vs_buy_hold"] > 0).any())
    upside_row = strategy_metrics.loc[strategy_metrics["strategy"] == UPSIDE_STRATEGY_NAME]
    upside = upside_row.iloc[0] if not upside_row.empty else None
    selected_signal = upside_signal_selection["selected"]
    selected_stability = signal_summary.loc[
        (signal_summary["model_name"] == selected_signal["model_name"])
        & (signal_summary["horizon"] == int(selected_signal["horizon"]))
        & (signal_summary["signal_name"] == selected_signal["signal_name"])
    ]
    selected_validation_stability = selected_stability.loc[selected_stability["period"] == "validation"]
    selected_test_stability = selected_stability.loc[selected_stability["period"] == "test"]
    validation_spread = (
        selected_validation_stability.iloc[0]["top_bottom_spread"]
        if not selected_validation_stability.empty
        else None
    )
    test_spread = selected_test_stability.iloc[0]["top_bottom_spread"] if not selected_test_stability.empty else None
    upside_outperforms = bool(_strategy_value(upside, "excess_return_vs_buy_hold") is not None and _strategy_value(upside, "excess_return_vs_buy_hold") > 0)
    upside_is_best_real = bool(best_real is not None and _strategy_value(best_real, "strategy") == UPSIDE_STRATEGY_NAME)
    stability_row = strategy_metrics.loc[strategy_metrics["strategy"] == STABILITY_AWARE_STRATEGY_NAME]
    stability_strategy = stability_row.iloc[0] if not stability_row.empty else None
    stability_outperforms = bool(
        _strategy_value(stability_strategy, "excess_return_vs_buy_hold") is not None
        and _strategy_value(stability_strategy, "excess_return_vs_buy_hold") > 0
    )
    stability_is_best_real = bool(best_real is not None and _strategy_value(best_real, "strategy") == STABILITY_AWARE_STRATEGY_NAME)
    risk_budget_row = strategy_metrics.loc[strategy_metrics["strategy"] == RISK_BUDGET_STRATEGY_NAME]
    risk_budget = risk_budget_row.iloc[0] if not risk_budget_row.empty else None
    risk_budget_outperforms = bool(
        _strategy_value(risk_budget, "excess_return_vs_buy_hold") is not None
        and _strategy_value(risk_budget, "excess_return_vs_buy_hold") > 0
    )
    risk_budget_drawdown_improvement = (
        _strategy_value(risk_budget, "max_drawdown") - _strategy_value(buy_hold, "max_drawdown")
        if risk_budget is not None and buy_hold is not None
        else None
    )
    risk_budget_reduces_drawdown = bool(
        risk_budget_drawdown_improvement is not None and risk_budget_drawdown_improvement > 0
    )
    risk_budget_is_best_real = bool(best_real is not None and _strategy_value(best_real, "strategy") == RISK_BUDGET_STRATEGY_NAME)
    early_stress_row = strategy_metrics.loc[strategy_metrics["strategy"] == EARLY_STRESS_STRATEGY_NAME]
    early_stress = early_stress_row.iloc[0] if not early_stress_row.empty else None
    early_stress_outperforms = bool(
        _strategy_value(early_stress, "excess_return_vs_buy_hold") is not None
        and _strategy_value(early_stress, "excess_return_vs_buy_hold") > 0
    )
    early_stress_drawdown_improvement = (
        _strategy_value(early_stress, "max_drawdown") - _strategy_value(buy_hold, "max_drawdown")
        if early_stress is not None and buy_hold is not None
        else None
    )
    early_stress_reduces_drawdown = bool(
        early_stress_drawdown_improvement is not None and early_stress_drawdown_improvement > 0
    )
    early_stress_is_best_real = bool(best_real is not None and _strategy_value(best_real, "strategy") == EARLY_STRESS_STRATEGY_NAME)
    early_event = drawdown_event_report["summary_by_strategy"].get(EARLY_STRESS_STRATEGY_NAME, {})
    early_reduced_before_trough = bool(early_event.get("reduced_exposure_before_trough", False))
    external_rows = final_strategy_comparison.loc[
        final_strategy_comparison["strategy"].astype(str).str.startswith("ml_external_")
    ].copy()
    best_external_return = (
        external_rows.sort_values("total_return", ascending=False).iloc[0]
        if not external_rows.empty
        else None
    )
    best_external_drawdown = (
        external_rows.sort_values("max_drawdown", ascending=False).iloc[0]
        if not external_rows.empty
        else None
    )
    soft_rows = external_rows.loc[external_rows["strategy"].astype(str).str.startswith("ml_external_soft_confirm")].copy()
    best_soft_return = (
        soft_rows.sort_values("total_return", ascending=False).iloc[0]
        if not soft_rows.empty
        else None
    )
    best_soft_drawdown = (
        soft_rows.sort_values("max_drawdown", ascending=False).iloc[0]
        if not soft_rows.empty
        else None
    )

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

    print("\n===== UPSIDE PARTICIPATION OPTIMIZATION RESULTS =====")
    print(
        "Selected upside signal model/horizon/signal_name: "
        f"{selected_signal['model_name']} / {selected_signal['horizon']}D / {selected_signal['signal_name']}"
    )
    print(f"Validation top-bottom spread: {fmt(validation_spread)}")
    print(f"Test top-bottom spread: {fmt(test_spread)}")
    print(f"Upside participation selected params: {upside_tuned['selected_params'].to_dict()}")
    print(f"Upside participation total return: {fmt(_strategy_value(upside, 'total_return'))}")
    print(f"Upside participation excess return: {fmt(_strategy_value(upside, 'excess_return_vs_buy_hold'))}")
    print(f"Upside participation max drawdown: {fmt(_strategy_value(upside, 'max_drawdown'))}")
    print(f"Upside participation avg position: {fmt(_strategy_value(upside, 'average_position'))}")
    print(f"Upside participation days below full exposure: {_strategy_value(upside, 'days_below_full_exposure')}")
    print(f"Buy-and-hold total return: {fmt(_strategy_value(buy_hold, 'total_return'))}")
    print(f"Whether upside participation outperforms buy-and-hold: {'Yes' if upside_outperforms else 'No'}")
    print(f"Whether it becomes best real no-leverage ML timing strategy: {'Yes' if upside_is_best_real else 'No'}")
    print("Output paths:")
    print(f"- {OUTPUT_DIR / 'signal_quantile_stability_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'upside_signal_model_selection.json'}")
    print(f"- {OUTPUT_DIR / 'upside_participation_tuned_params.json'}")
    print(f"- {OUTPUT_DIR / 'upside_participation_attribution_summary.md'}")
    print(f"- {OUTPUT_DIR / 'summary.md'}")
    print(f"- {OUTPUT_DIR / 'final_report_material.md'}")
    print("=====================================================")

    print("\n===== STABILITY-AWARE HIGH-PARTICIPATION RESULTS =====")
    print("Stability-aware selected components:")
    for idx, component in enumerate(selected_components, start=1):
        print(
            f"- {idx}: {component['model_name']} / {component['horizon']}D / "
            f"{component['signal_name']} / stable_score={fmt(component.get('stable_score'))}"
        )
    print(f"Stability-aware selected params: {stability_tuned['selected_params'].to_dict()}")
    print(f"Stability-aware total return: {fmt(_strategy_value(stability_strategy, 'total_return'))}")
    print(f"Stability-aware excess return: {fmt(_strategy_value(stability_strategy, 'excess_return_vs_buy_hold'))}")
    print(f"Stability-aware max drawdown: {fmt(_strategy_value(stability_strategy, 'max_drawdown'))}")
    print(f"Stability-aware avg position: {fmt(_strategy_value(stability_strategy, 'average_position'))}")
    print(f"Stability-aware days below full exposure: {_strategy_value(stability_strategy, 'days_below_full_exposure')}")
    print(f"Buy-and-hold total return: {fmt(_strategy_value(buy_hold, 'total_return'))}")
    print(f"Whether stability-aware strategy outperforms buy-and-hold: {'Yes' if stability_outperforms else 'No'}")
    print(f"Whether it becomes best real no-leverage ML timing strategy: {'Yes' if stability_is_best_real else 'No'}")
    print("Output paths:")
    print(f"- {OUTPUT_DIR / 'stability_aware_signal_selection.json'}")
    print(f"- {OUTPUT_DIR / 'stability_aware_upside_score.csv'}")
    print(f"- {OUTPUT_DIR / 'stability_aware_high_participation_tuned_params.json'}")
    print(f"- {OUTPUT_DIR / 'stability_aware_high_participation_attribution_summary.md'}")
    print(f"- {PLOT_DIR / 'equity_curve_stability_aware_high_participation_vs_buyhold.png'}")
    print(f"- {PLOT_DIR / 'drawdown_curve_stability_aware_high_participation_vs_buyhold.png'}")
    print(f"- {PLOT_DIR / 'position_exposure_stability_aware_high_participation.png'}")
    print(f"- {PLOT_DIR / 'stability_aware_signal_diagnostics.png'}")
    print(f"- {PLOT_DIR / 'stability_aware_component_scores.png'}")
    print("======================================================")

    print("\n===== RISK-BUDGET DRAWDOWN CONTROL RESULTS =====")
    print(f"Risk-budget selected params: {risk_budget_tuned['selected_params'].to_dict()}")
    print(f"Risk-budget total return: {fmt(_strategy_value(risk_budget, 'total_return'))}")
    print(f"Risk-budget excess return: {fmt(_strategy_value(risk_budget, 'excess_return_vs_buy_hold'))}")
    print(f"Risk-budget max drawdown: {fmt(_strategy_value(risk_budget, 'max_drawdown'))}")
    print(f"Risk-budget drawdown improvement vs buy-and-hold: {fmt(risk_budget_drawdown_improvement)}")
    print(f"Risk-budget avg position: {fmt(_strategy_value(risk_budget, 'average_position'))}")
    print(f"Risk-budget days below full exposure: {_strategy_value(risk_budget, 'days_below_full_exposure')}")
    print(f"Buy-and-hold total return: {fmt(_strategy_value(buy_hold, 'total_return'))}")
    print(f"Buy-and-hold max drawdown: {fmt(_strategy_value(buy_hold, 'max_drawdown'))}")
    print(f"Whether risk-budget outperforms buy-and-hold: {'Yes' if risk_budget_outperforms else 'No'}")
    print(f"Whether risk-budget reduces max drawdown: {'Yes' if risk_budget_reduces_drawdown else 'No'}")
    print(f"Whether it becomes best real no-leverage ML timing strategy: {'Yes' if risk_budget_is_best_real else 'No'}")
    print("Output paths:")
    print(f"- {OUTPUT_DIR / 'risk_budget_tuned_params.json'}")
    print(f"- {OUTPUT_DIR / 'risk_budget_attribution_summary.md'}")
    print(f"- {PLOT_DIR / 'equity_curve_risk_budget_vs_buyhold.png'}")
    print(f"- {PLOT_DIR / 'drawdown_curve_risk_budget_vs_buyhold.png'}")
    print(f"- {PLOT_DIR / 'position_exposure_risk_budget.png'}")
    print(f"- {PLOT_DIR / 'risk_budget_signal_diagnostics.png'}")
    print("=================================================")

    print("\n===== EARLY-STRESS RISK-BUDGET RESULTS =====")
    print(f"Early-stress selected params: {early_stress_tuned['selected_params'].to_dict()}")
    print(f"Early-stress total return: {fmt(_strategy_value(early_stress, 'total_return'))}")
    print(f"Early-stress excess return: {fmt(_strategy_value(early_stress, 'excess_return_vs_buy_hold'))}")
    print(f"Early-stress max drawdown: {fmt(_strategy_value(early_stress, 'max_drawdown'))}")
    print(f"Early-stress drawdown improvement: {fmt(early_stress_drawdown_improvement)}")
    print(f"Early-stress avg position: {fmt(_strategy_value(early_stress, 'average_position'))}")
    print(f"Early-stress days below full exposure: {_strategy_value(early_stress, 'days_below_full_exposure')}")
    print(f"Whether early-stress outperforms buy-and-hold: {'Yes' if early_stress_outperforms else 'No'}")
    print(f"Whether early-stress reduces max drawdown: {'Yes' if early_stress_reduces_drawdown else 'No'}")
    print(f"Whether early-stress becomes best real no-leverage ML strategy: {'Yes' if early_stress_is_best_real else 'No'}")
    print(f"Whether early-stress reduced exposure before benchmark trough: {'Yes' if early_reduced_before_trough else 'No'}")
    print("Output paths:")
    print(f"- {OUTPUT_DIR / 'early_stress_risk_budget_tuned_params.json'}")
    print(f"- {OUTPUT_DIR / 'early_stress_risk_budget_attribution_summary.md'}")
    print(f"- {PLOT_DIR / 'equity_curve_early_stress_risk_budget_vs_buyhold.png'}")
    print(f"- {PLOT_DIR / 'drawdown_curve_early_stress_risk_budget_vs_buyhold.png'}")
    print(f"- {PLOT_DIR / 'position_exposure_early_stress_risk_budget.png'}")
    print(f"- {PLOT_DIR / 'early_stress_signal_diagnostics.png'}")
    print("================================================")

    print("\n===== EXTERNAL MARKET DATA ABLATION RESULTS =====")
    print(f"Best external strategy by total return: {_strategy_value(best_external_return, 'strategy')}")
    print(f"Best external total return: {fmt(_strategy_value(best_external_return, 'total_return'))}")
    print(f"Best external excess return: {fmt(_strategy_value(best_external_return, 'excess_return_vs_buy_hold'))}")
    print(f"Best external strategy by max drawdown: {_strategy_value(best_external_drawdown, 'strategy')}")
    print(f"Best external max drawdown: {fmt(_strategy_value(best_external_drawdown, 'max_drawdown'))}")
    print(f"Best soft external strategy by total return: {_strategy_value(best_soft_return, 'strategy')}")
    print(f"Best soft external total return: {fmt(_strategy_value(best_soft_return, 'total_return'))}")
    print(f"Best soft external excess return: {fmt(_strategy_value(best_soft_return, 'excess_return_vs_buy_hold'))}")
    print(f"Best soft external strategy by max drawdown: {_strategy_value(best_soft_drawdown, 'strategy')}")
    print(f"Best soft external max drawdown: {fmt(_strategy_value(best_soft_drawdown, 'max_drawdown'))}")
    print("Output paths:")
    print(f"- {OUTPUT_DIR / 'final_strategy_comparison.csv'}")
    print(f"- {OUTPUT_DIR / 'final_strategy_comparison.md'}")
    print(f"- {OUTPUT_DIR / 'external_market_feature_report.md'}")
    print(f"- {OUTPUT_DIR / 'model_stability_audit.md'}")
    print(f"- {OUTPUT_DIR / 'model_strategy_utility_comparison.md'}")
    print(f"- {OUTPUT_DIR / 'current' / 'final_strategy_comparison.csv'}")
    print(f"- {OUTPUT_DIR / 'current' / 'external_market_feature_report.md'}")
    print(f"- {OUTPUT_DIR / 'current' / 'model_stability_audit.md'}")
    print(f"- {OUTPUT_DIR / 'current' / 'model_strategy_utility_comparison.md'}")
    print("=================================================")
    print_drawdown_event_diagnostics(drawdown_event_report)


if __name__ == "__main__":
    main()
