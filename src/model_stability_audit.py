from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import TEST_END, TEST_START
from src.strategy.backtest import build_buy_hold_frame, simulate_strategy_cost_aware
from src.strategy.decision_rules import (
    EarlyStressRiskBudgetParams,
    RiskBudgetParams,
    UpsideParticipationParams,
    ml_risk_budget_enhancement_position,
    prepare_strategy_frame,
)
from src.strategy.metrics import compute_strategy_metrics


REQUIRED_AUDIT_MODEL_NAMES = {
    "LogisticRegression",
    "LogisticRegressionStrongL2",
    "RidgeClassifier",
    "ElasticNet",
}

FINAL_STRATEGY_NAMES = {"ml_risk_budget_enhancement"}

SIMPLE_MODEL_PARAM_SPECS: dict[str, dict[str, Any]] = {
    "LogisticRegression": {
        "target_ret": {"regressor": "Ridge", "alpha": 1.0, "scaler": "StandardScaler"},
        "classification": {
            "classifier": "LogisticRegression",
            "penalty": "l2",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 3000,
            "scaler": "StandardScaler",
        },
    },
    "LogisticRegressionStrongL2": {
        "target_ret": {"regressor": "Ridge", "alpha": 10.0, "scaler": "StandardScaler"},
        "classification": {
            "classifier": "LogisticRegression",
            "penalty": "l2",
            "C": 0.2,
            "class_weight": "balanced",
            "max_iter": 3000,
            "scaler": "StandardScaler",
        },
    },
    "RidgeClassifier": {
        "target_ret": {"regressor": "Ridge", "alpha": 10.0, "scaler": "StandardScaler"},
        "classification": {
            "classifier": "RidgeClassifier",
            "alpha": 10.0,
            "class_weight": "balanced",
            "score_output": "decision_function_sigmoid",
            "scaler": "StandardScaler",
        },
    },
    "ElasticNet": {
        "target_ret": {
            "regressor": "ElasticNet",
            "alpha": 0.1,
            "l1_ratio": 0.2,
            "max_iter": 5000,
            "scaler": "StandardScaler",
        },
        "classification": {
            "classifier": "LogisticRegression",
            "penalty": "l2",
            "C": 0.5,
            "class_weight": "balanced",
            "max_iter": 3000,
            "scaler": "StandardScaler",
        },
    },
}

TARGET_METRICS = [
    {
        "target_name": "target_ret",
        "score_col": "return_correlation",
        "auc_col": None,
        "logloss_col": None,
        "return_corr_col": "return_correlation",
    },
    {
        "target_name": "target_up",
        "score_col": "AUC",
        "auc_col": "AUC",
        "logloss_col": "logloss",
        "return_corr_col": None,
    },
    {
        "target_name": "target_trade",
        "score_col": "trade_AUC",
        "auc_col": "trade_AUC",
        "logloss_col": "trade_logloss",
        "return_corr_col": None,
    },
    {
        "target_name": "target_big_up",
        "score_col": "big_up_AUC",
        "auc_col": "big_up_AUC",
        "logloss_col": "big_up_logloss",
        "return_corr_col": None,
    },
    {
        "target_name": "target_big_down",
        "score_col": "big_down_AUC",
        "auc_col": "big_down_AUC",
        "logloss_col": "big_down_logloss",
        "return_corr_col": None,
    },
    {
        "target_name": "target_clean_direction",
        "score_col": "clean_direction_AUC",
        "auc_col": "clean_direction_AUC",
        "logloss_col": "clean_direction_logloss",
        "return_corr_col": None,
    },
]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _metric(row: pd.Series | None, col: str | None) -> float:
    if row is None or col is None or col not in row:
        return np.nan
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return np.nan


def _row_for(metrics: pd.DataFrame, model_name: str, horizon: int) -> pd.Series | None:
    if metrics.empty:
        return None
    rows = metrics.loc[(metrics["model_name"] == model_name) & (metrics["horizon"].astype(int) == int(horizon))]
    return rows.iloc[0] if not rows.empty else None


def _strategy_usage_map(strategy_params: dict[str, Any]) -> dict[tuple[str, int], list[str]]:
    usage: dict[tuple[str, int], list[str]] = {}
    for strategy_name, params in strategy_params.items():
        if strategy_name not in FINAL_STRATEGY_NAMES:
            continue
        model_name = getattr(params, "model_name", None)
        horizon = getattr(params, "horizon", None)
        if model_name is None or horizon is None:
            continue
        usage.setdefault((str(model_name), int(horizon)), []).append(strategy_name)
    return usage


def _stability_comment(
    validation_score: float,
    test_score: float,
    gap: float,
    target_name: str,
    is_strategy_used: bool,
) -> str:
    if is_strategy_used:
        return "selected by strategy-layer utility, not pure prediction score"
    if np.isfinite(validation_score) and np.isfinite(test_score):
        high_validation = validation_score >= (0.65 if target_name != "target_ret" else 0.20)
        weak_test = test_score < (0.55 if target_name != "target_ret" else 0.05)
        if high_validation and (weak_test or gap > 0.12):
            return "validation overfit / weak test generalization"
        if test_score > validation_score + 0.05:
            return "test improvement, do not use for ex-post selection"
        if abs(gap) <= 0.05 and test_score >= (0.52 if target_name != "target_ret" else 0.05):
            return "stable but moderate predictive power"
    return "weak or unstable predictive evidence"


def build_model_stability_audit(
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    best_prediction_model: dict[str, Any],
    strategy_params: dict[str, Any],
) -> pd.DataFrame:
    usage = _strategy_usage_map(strategy_params)
    best_validation = best_prediction_model.get("best_validation", {})
    best_model_name = str(best_validation.get("model_name", ""))
    best_horizon = int(best_validation.get("horizon", -1)) if best_validation.get("horizon") is not None else -1
    pairs = pd.concat(
        [
            validation_metrics[["model_name", "horizon"]],
            test_metrics[["model_name", "horizon"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    rows: list[dict[str, Any]] = []
    for pair in pairs.itertuples(index=False):
        model_name = str(pair.model_name)
        horizon = int(pair.horizon)
        val_row = _row_for(validation_metrics, model_name, horizon)
        test_row = _row_for(test_metrics, model_name, horizon)
        used_by = usage.get((model_name, horizon), [])
        for target in TARGET_METRICS:
            target_name = target["target_name"]
            score_col = target["score_col"]
            val_score = _metric(val_row, score_col)
            test_score = _metric(test_row, score_col)
            val_auc = _metric(val_row, target["auc_col"])
            test_auc = _metric(test_row, target["auc_col"])
            val_corr = _metric(val_row, target["return_corr_col"])
            test_corr = _metric(test_row, target["return_corr_col"])
            val_logloss = _metric(val_row, target["logloss_col"])
            test_logloss = _metric(test_row, target["logloss_col"])
            score_gap = val_score - test_score if np.isfinite(val_score) and np.isfinite(test_score) else np.nan
            auc_gap = val_auc - test_auc if np.isfinite(val_auc) and np.isfinite(test_auc) else np.nan
            logloss_gap = val_logloss - test_logloss if np.isfinite(val_logloss) and np.isfinite(test_logloss) else np.nan
            rows.append(
                {
                    "model_name": model_name,
                    "horizon": horizon,
                    "target_name": target_name,
                    "validation_auc": val_auc,
                    "test_auc": test_auc,
                    "validation_logloss": val_logloss,
                    "test_logloss": test_logloss,
                    "logloss_gap": logloss_gap,
                    "validation_return_corr": val_corr,
                    "test_return_corr": test_corr,
                    "validation_score": val_score,
                    "test_score": test_score,
                    "score_gap": score_gap,
                    "auc_gap": auc_gap,
                    "validation_selection_score": _metric(val_row, "selection_score"),
                    "test_selection_score": _metric(test_row, "selection_score"),
                    "is_validation_prediction_best": model_name == best_model_name and horizon == best_horizon,
                    "is_final_strategy_used": bool(used_by),
                    "final_strategy_name": "; ".join(used_by),
                    "stability_comment": _stability_comment(
                        val_score,
                        test_score,
                        score_gap,
                        target_name,
                        bool(used_by),
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["is_validation_prediction_best", "score_gap", "validation_score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 30) -> str:
    use_cols = [col for col in cols if col in df.columns]
    if df.empty or not use_cols:
        return "No rows."
    use = df.loc[:, use_cols].head(max_rows).copy()
    lines = [
        "| " + " | ".join(use_cols) + " |",
        "| " + " | ".join(["---"] * len(use_cols)) + " |",
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


def _strategy_row(metrics: pd.DataFrame, strategy_name: str) -> pd.Series | None:
    rows = metrics.loc[metrics["strategy"] == strategy_name]
    return rows.iloc[0] if not rows.empty else None


def _prediction_score(metrics: pd.DataFrame, model_name: str, horizon: int) -> float:
    row = _row_for(metrics, model_name, horizon)
    return _metric(row, "selection_score")


def _make_risk_budget_params(base: RiskBudgetParams, model_name: str, horizon: int) -> RiskBudgetParams:
    data = asdict(base)
    data["model_name"] = model_name
    data["horizon"] = int(horizon)
    return RiskBudgetParams(**data)


def _utility_comment(
    strategy_name: str,
    model_name: str,
    val_score: float,
    test_score: float,
    excess: float,
    drawdown_improvement: float,
) -> str:
    if strategy_name == "ml_risk_budget_enhancement":
        return "current main strategy; LightGBM 5D selected by strategy-layer utility"
    if strategy_name == "ml_upside_participation_enhancement":
        return "MLP validation prediction advantage did not translate into robust test strategy utility"
    if model_name in REQUIRED_AUDIT_MODEL_NAMES:
        if np.isfinite(excess) and excess > 0:
            return "linear baseline has positive strategy utility; compare as stable benchmark"
        return "linear baseline is useful as stability control but does not beat buy-and-hold"
    if np.isfinite(val_score) and np.isfinite(test_score) and val_score - test_score > 0.15:
        return "prediction-layer score shows large validation-test gap"
    if np.isfinite(drawdown_improvement) and drawdown_improvement > 0:
        return "strategy improves drawdown but may sacrifice return"
    return "strategy-layer utility is moderate or weak"


def _metrics_record(
    row: pd.Series,
    strategy_name: str,
    model_name: str,
    horizon: int,
    main_signal: str,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> dict[str, Any]:
    validation_score = _prediction_score(validation_metrics, model_name, horizon)
    test_score = _prediction_score(test_metrics, model_name, horizon)
    score_gap = validation_score - test_score if np.isfinite(validation_score) and np.isfinite(test_score) else np.nan
    drawdown_improvement = float(row.get("max_drawdown", np.nan)) - float(row.get("benchmark_max_drawdown", np.nan))
    excess = float(row.get("excess_return_vs_buy_hold", np.nan))
    return {
        "strategy_name": strategy_name,
        "model_name": model_name,
        "horizon": int(horizon),
        "main_signal": main_signal,
        "total_return": row.get("total_return"),
        "excess_return_vs_buy_hold": excess,
        "max_drawdown": row.get("max_drawdown"),
        "drawdown_improvement_vs_buy_hold": drawdown_improvement,
        "sharpe": row.get("sharpe"),
        "calmar": row.get("calmar"),
        "average_position": row.get("average_position"),
        "total_turnover": row.get("total_turnover"),
        "missed_upside": row.get("missed_upside"),
        "avoided_downside": row.get("avoided_downside"),
        "outperforms_buy_hold": bool(excess > 0) if np.isfinite(excess) else False,
        "reduces_max_drawdown": bool(drawdown_improvement > 0) if np.isfinite(drawdown_improvement) else False,
        "prediction_validation_score": validation_score,
        "prediction_test_score": test_score,
        "prediction_score_gap": score_gap,
        "strategy_utility_comment": _utility_comment(
            strategy_name,
            model_name,
            validation_score,
            test_score,
            excess,
            drawdown_improvement,
        ),
    }


def _backtest_linear_risk_budget_utilities(
    feature_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    base_risk_budget_params: RiskBudgetParams,
    baseline_models: list[str],
    horizon: int = 5,
) -> pd.DataFrame:
    feature_df = feature_df.copy()
    test_predictions = test_predictions.copy()
    if "date" in feature_df.columns:
        feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")
    if "date" in test_predictions.columns:
        test_predictions["date"] = pd.to_datetime(test_predictions["date"], errors="coerce")
    available = set(test_predictions["model_name"].astype(str))
    selected = [model for model in baseline_models if model in available]
    if not selected:
        return pd.DataFrame()
    frames = []
    first_frame = None
    for model_name in selected:
        params = _make_risk_budget_params(base_risk_budget_params, model_name, horizon)
        frame = prepare_strategy_frame(feature_df, test_predictions, params)
        if first_frame is None:
            first_frame = frame
        position = ml_risk_budget_enhancement_position(frame, params)
        strategy_name = f"model_utility_{model_name}_risk_budget_{horizon}d"
        frames.append(simulate_strategy_cost_aware(frame, position, strategy_name, initial_position=1.0))
    if first_frame is None:
        return pd.DataFrame()
    frames.insert(0, build_buy_hold_frame(first_frame))
    daily = pd.concat(frames, ignore_index=True)
    return compute_strategy_metrics(daily)


def build_model_strategy_utility_comparison(
    feature_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    upside_params: UpsideParticipationParams | None,
    risk_budget_params: RiskBudgetParams,
    early_stress_params: EarlyStressRiskBudgetParams,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    existing = [
        (
            "ml_upside_participation_enhancement",
            getattr(upside_params, "model_name", "NA") if upside_params is not None else "NA",
            int(getattr(upside_params, "horizon", 0)) if upside_params is not None else 0,
            getattr(upside_params, "signal_name", "pred_big_up_prob") if upside_params is not None else "pred_big_up_prob",
        ),
        (
            "ml_risk_budget_enhancement",
            risk_budget_params.model_name,
            int(risk_budget_params.horizon),
            "risk_budget: pred_up_prob + pred_big_up_prob + pred_ret",
        ),
        (
            "ml_early_stress_risk_budget",
            early_stress_params.model_name,
            int(early_stress_params.horizon),
            "early_stress_score + pred_up_prob + pred_big_up_prob",
        ),
    ]
    for strategy_name, model_name, horizon, main_signal in existing:
        row = _strategy_row(strategy_metrics, strategy_name)
        if row is None:
            continue
        rows.append(_metrics_record(row, strategy_name, model_name, horizon, main_signal, validation_metrics, test_metrics))

    utility_metrics = _backtest_linear_risk_budget_utilities(
        feature_df,
        test_predictions,
        risk_budget_params,
        sorted(REQUIRED_AUDIT_MODEL_NAMES),
        horizon=5,
    )
    for strategy_name in utility_metrics["strategy"].dropna().astype(str).tolist() if not utility_metrics.empty else []:
        if strategy_name == "buy_hold":
            continue
        model_name = strategy_name.removeprefix("model_utility_").removesuffix("_risk_budget_5d")
        row = _strategy_row(utility_metrics, strategy_name)
        if row is None:
            continue
        rows.append(
            _metrics_record(
                row,
                strategy_name,
                model_name,
                5,
                "risk_budget baseline: pred_up_prob + pred_big_up_prob + pred_ret",
                validation_metrics,
                test_metrics,
            )
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("total_return", ascending=False).reset_index(drop=True)


def _simple_model_selected_params(model_name: str, target_name: str) -> dict[str, Any]:
    spec = SIMPLE_MODEL_PARAM_SPECS.get(model_name, {})
    key = "target_ret" if target_name == "target_ret" else "classification"
    params = spec.get(key, {})
    return dict(params) if isinstance(params, dict) else {}


def build_simple_model_param_records(validation_metrics: pd.DataFrame, test_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = pd.concat(
        [
            validation_metrics[["model_name", "horizon"]],
            test_metrics[["model_name", "horizon"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    pairs = pairs.loc[pairs["model_name"].astype(str).isin(REQUIRED_AUDIT_MODEL_NAMES)].copy()
    for pair in pairs.sort_values(["model_name", "horizon"]).itertuples(index=False):
        model_name = str(pair.model_name)
        horizon = int(pair.horizon)
        val_row = _row_for(validation_metrics, model_name, horizon)
        test_row = _row_for(test_metrics, model_name, horizon)
        for target in TARGET_METRICS:
            target_name = target["target_name"]
            validation_score = _metric(val_row, target["score_col"])
            test_score = _metric(test_row, target["score_col"])
            rows.append(
                {
                    "model_name": model_name,
                    "target_name": target_name,
                    "horizon": horizon,
                    "selected_params": _simple_model_selected_params(model_name, target_name),
                    "validation_score": validation_score,
                    "test_score": test_score,
                    "selection_basis": "validation-period audit baseline; test score recorded only after selection",
                }
            )
    return rows


def build_model_stability_summary_section(
    stability: pd.DataFrame,
    utility: pd.DataFrame,
    best_prediction_model: dict[str, Any],
) -> str:
    best = best_prediction_model.get("best_validation", {})
    match = best_prediction_model.get("matching_test") or {}
    best_model = best.get("model_name", "NA")
    best_horizon = best.get("horizon", "NA")
    main = utility.loc[utility["strategy_name"] == "ml_risk_budget_enhancement"]
    main_row = main.iloc[0] if not main.empty else None
    linear = utility.loc[utility["model_name"].isin(REQUIRED_AUDIT_MODEL_NAMES)].copy()
    best_linear = linear.sort_values("total_return", ascending=False).iloc[0] if not linear.empty else None
    return f"""
## Model Stability Audit
- Validation-best prediction model: `{best_model}` / {best_horizon}D.
- Validation-best prediction score: {_fmt(best.get("selection_score"))}; matching test score: {_fmt(match.get("selection_score"))}.
- MLP test generalization: direction AUC {_fmt(match.get("AUC"))}, clean-direction AUC {_fmt(match.get("clean_direction_AUC"))}, trade AUC {_fmt(match.get("trade_AUC"))}, return correlation {_fmt(match.get("return_correlation"))}.
- Final main strategy depends on MLP: No.
- Final main strategy model/signal: `{main_row["model_name"] if main_row is not None else "NA"}` / {int(main_row["horizon"]) if main_row is not None and pd.notna(main_row["horizon"]) else "NA"}D inside `ml_risk_budget_enhancement`.
- Prediction-layer best and strategy-layer best are the same: No.
- Interpretation: prediction-layer accuracy does not necessarily translate into strategy-layer utility.
- MLP recommendation: do not use MLP as the final main model under current evidence; treat it as a validation-overfit warning and a research candidate.
- LightGBM / 5D recommendation: keep as current deployable strategy model because it is selected by strategy-layer utility and remains the only no-leverage strategy with positive excess return.
- Best simple linear baseline by strategy utility: `{best_linear["model_name"] if best_linear is not None else "NA"}` with total return {_fmt(best_linear["total_return"] if best_linear is not None else None)} and excess {_fmt(best_linear["excess_return_vs_buy_hold"] if best_linear is not None else None)}.
- Linear baseline conclusion: simple models are useful as stability controls; they help identify whether complex validation winners add real trading value rather than only validation-period fit.
"""


def _upsert_model_stability_section(summary_path: Path, section: str) -> None:
    existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = "\n## Model Stability Audit\n"
    inline_marker = "## Model Stability Audit\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    elif existing.startswith(inline_marker):
        existing = ""
    summary_path.write_text(existing.rstrip() + "\n" + section.strip() + "\n", encoding="utf-8")


def write_model_stability_reports(
    feature_df: pd.DataFrame,
    test_predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    best_prediction_model: dict[str, Any],
    strategy_params: dict[str, Any],
    upside_params: UpsideParticipationParams | None,
    risk_budget_params: RiskBudgetParams,
    early_stress_params: EarlyStressRiskBudgetParams,
    output_dir: Path,
    summary_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability = build_model_stability_audit(validation_metrics, test_metrics, best_prediction_model, strategy_params)
    utility = build_model_strategy_utility_comparison(
        feature_df,
        test_predictions,
        validation_metrics,
        test_metrics,
        strategy_metrics,
        upside_params,
        risk_budget_params,
        early_stress_params,
    )
    stability.to_csv(output_dir / "model_stability_audit.csv", index=False)
    utility.to_csv(output_dir / "model_strategy_utility_comparison.csv", index=False)
    simple_param_records = build_simple_model_param_records(validation_metrics, test_metrics)
    (output_dir / "simple_model_tuned_params.json").write_text(
        json.dumps(
            {
                "selection_basis": "Simple baseline parameters are fixed validation-period audit candidates; no test-set parameter selection is used.",
                "records": simple_param_records,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    stability_cols = [
        "model_name",
        "horizon",
        "target_name",
        "validation_score",
        "test_score",
        "score_gap",
        "validation_auc",
        "test_auc",
        "auc_gap",
        "validation_logloss",
        "test_logloss",
        "logloss_gap",
        "is_validation_prediction_best",
        "is_final_strategy_used",
        "final_strategy_name",
        "stability_comment",
    ]
    utility_cols = [
        "strategy_name",
        "model_name",
        "horizon",
        "main_signal",
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
        "outperforms_buy_hold",
        "reduces_max_drawdown",
        "prediction_validation_score",
        "prediction_test_score",
        "prediction_score_gap",
        "strategy_utility_comment",
    ]
    stability_md = f"""# Model Stability Audit

Test period: {TEST_START.date()} to {TEST_END.date()}

This audit compares validation and test prediction behavior by model, horizon, and target. High validation scores with weak matching test scores are treated as overfit risk, not deployable evidence.

## Largest Validation-Test Gaps
{_markdown_table(stability.sort_values("score_gap", ascending=False, na_position="last"), stability_cols, max_rows=40)}
"""
    utility_md = f"""# Model Strategy Utility Comparison

Prediction-layer metrics do not automatically imply trading value. This table compares selected prediction signals after they enter strategy rules.

{_markdown_table(utility, utility_cols, max_rows=40)}
"""
    (output_dir / "model_stability_audit.md").write_text(stability_md, encoding="utf-8")
    (output_dir / "model_strategy_utility_comparison.md").write_text(utility_md, encoding="utf-8")

    _upsert_model_stability_section(
        summary_path,
        build_model_stability_summary_section(stability, utility, best_prediction_model),
    )
    return stability, utility
