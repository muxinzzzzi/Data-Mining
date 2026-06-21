from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import INITIAL_CAPITAL, VALID_END, VALID_START
from src.signal_stability import build_stability_aware_upside_score
from src.strategy.backtest import simulate_strategy_cost_aware
from src.strategy.decision_rules import (
    EarlyStressRiskBudgetParams,
    ExternalEarlyStressRiskBudgetParams,
    ExternalRiskBudgetParams,
    ExternalSoftConfirmEarlyStressParams,
    ExternalSoftConfirmRiskBudgetParams,
    RiskBudgetParams,
    StabilityAwareHighParticipationParams,
    UpsideParticipationParams,
    ml_early_stress_risk_budget_position,
    ml_external_soft_lite_early_stress_position,
    ml_external_soft_lite_risk_budget_position,
    ml_external_soft_confirm_early_stress_position,
    ml_external_soft_confirm_risk_budget_position,
    ml_risk_budget_enhancement_position,
    ml_stability_aware_high_participation_position,
    ml_upside_participation_enhancement_position,
    prepare_strategy_frame,
)
from src.utils import write_json_ready


UPSIDE_STRATEGY_NAME = "ml_upside_participation_enhancement"
STABILITY_AWARE_STRATEGY_NAME = "ml_stability_aware_high_participation"
RISK_BUDGET_STRATEGY_NAME = "ml_risk_budget_enhancement"
EARLY_STRESS_STRATEGY_NAME = "ml_early_stress_risk_budget"
EXTERNAL_RISK_BUDGET_STRATEGY_NAMES = {
    "index_only": "ml_external_risk_budget_enhancement_index_only",
    "index_margin": "ml_external_risk_budget_enhancement_index_margin",
    "index_shibor": "ml_external_risk_budget_enhancement_index_shibor",
    "all": "ml_external_risk_budget_enhancement_all",
}
EXTERNAL_EARLY_STRESS_STRATEGY_NAMES = {
    "index_only": "ml_external_early_stress_risk_budget_index_only",
    "all": "ml_external_early_stress_risk_budget_all",
}
EXTERNAL_SOFT_RISK_BUDGET_STRATEGY_NAMES = {
    "all": "ml_external_soft_confirm_risk_budget",
    "index_only": "ml_external_soft_confirm_risk_budget_index_only",
}
EXTERNAL_SOFT_EARLY_STRESS_STRATEGY_NAMES = {
    "all": "ml_external_soft_confirm_early_stress",
    "index_only": "ml_external_soft_confirm_early_stress_index_only",
}
EXTERNAL_SOFT_LITE_RISK_BUDGET_STRATEGY_NAMES = {
    "all": "ml_external_soft_lite_risk_budget",
    "index_only": "ml_external_soft_lite_risk_budget_index_only",
}
EXTERNAL_SOFT_LITE_EARLY_STRESS_STRATEGY_NAMES = {
    "all": "ml_external_soft_lite_early_stress",
    "index_only": "ml_external_soft_lite_early_stress_index_only",
}


def build_external_market_ablation_params(
    risk_budget_params: RiskBudgetParams,
    early_stress_params: EarlyStressRiskBudgetParams,
) -> dict[str, ExternalRiskBudgetParams | ExternalEarlyStressRiskBudgetParams]:
    out: dict[str, ExternalRiskBudgetParams | ExternalEarlyStressRiskBudgetParams] = {}
    for feature_set, strategy_name in EXTERNAL_RISK_BUDGET_STRATEGY_NAMES.items():
        out[strategy_name] = ExternalRiskBudgetParams(
            **risk_budget_params.to_dict(),
            external_feature_set=feature_set,
        )
    for feature_set, strategy_name in EXTERNAL_EARLY_STRESS_STRATEGY_NAMES.items():
        out[strategy_name] = ExternalEarlyStressRiskBudgetParams(
            **early_stress_params.to_dict(),
            external_feature_set=feature_set,
        )
    return out


def _soft_risk_param_from_base(
    base: RiskBudgetParams,
    feature_set: str,
    high_confirm_weight: float,
    extreme_confirm_weight: float,
    soft_min_weight: float,
    style_floor: float,
    require_persistence: bool,
    high_unconfirmed_weight: float = 0.98,
    extreme_unconfirmed_weight: float = 0.96,
    extreme_stress_weight: float = 0.85,
    strong_style_floor: float = 0.98,
) -> ExternalSoftConfirmRiskBudgetParams:
    return ExternalSoftConfirmRiskBudgetParams(
        **base.to_dict(),
        external_feature_set=feature_set,
        high_threshold=4.0,
        extreme_threshold=5.0,
        high_unconfirmed_weight=high_unconfirmed_weight,
        high_confirm_weight=high_confirm_weight,
        extreme_unconfirmed_weight=extreme_unconfirmed_weight,
        extreme_confirm_weight=extreme_confirm_weight,
        extreme_stress_weight=extreme_stress_weight,
        soft_min_weight=soft_min_weight,
        style_floor=style_floor,
        strong_style_floor=strong_style_floor,
        require_persistence=require_persistence,
        persistence_days=2,
    )


def _soft_early_param_from_base(
    base: EarlyStressRiskBudgetParams,
    feature_set: str,
    high_confirm_weight: float,
    extreme_confirm_weight: float,
    soft_min_weight: float,
    style_floor: float,
    require_persistence: bool,
    high_unconfirmed_weight: float = 0.98,
    extreme_unconfirmed_weight: float = 0.96,
    extreme_stress_weight: float = 0.85,
    strong_style_floor: float = 0.98,
) -> ExternalSoftConfirmEarlyStressParams:
    return ExternalSoftConfirmEarlyStressParams(
        **base.to_dict(),
        external_feature_set=feature_set,
        high_threshold=4.0,
        extreme_threshold=5.0,
        high_unconfirmed_weight=high_unconfirmed_weight,
        high_confirm_weight=high_confirm_weight,
        extreme_unconfirmed_weight=extreme_unconfirmed_weight,
        extreme_confirm_weight=extreme_confirm_weight,
        extreme_stress_weight=extreme_stress_weight,
        soft_min_weight=soft_min_weight,
        style_floor=style_floor,
        strong_style_floor=strong_style_floor,
        require_persistence=require_persistence,
        persistence_days=2,
    )


def _evaluate_soft_confirmation_frame(frame: pd.DataFrame, final_position: pd.Series) -> dict[str, float]:
    sim = simulate_strategy_cost_aware(
        frame.reset_index(drop=True),
        final_position.reset_index(drop=True),
        "validation_soft_confirmation",
        initial_position=1.0,
    )
    position = sim["final_position"].astype(float)
    turnover = sim["turnover"].astype(float)
    transaction_cost = sim["transaction_cost"].astype(float)
    strategy_return = sim["strategy_return"].astype(float)
    equity = sim["equity"].astype(float)
    buy_hold_equity = sim["buy_hold_equity"].astype(float)
    drawdown = sim["drawdown"].astype(float)
    buy_hold_drawdown = sim["buy_hold_drawdown"].astype(float)
    total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1) if len(equity) else np.nan
    benchmark_total = float(buy_hold_equity.iloc[-1] / INITIAL_CAPITAL - 1) if len(buy_hold_equity) else np.nan
    max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
    benchmark_max_drawdown = float(buy_hold_drawdown.min()) if len(buy_hold_drawdown) else np.nan
    ann_return = _annualized_return(equity)
    benchmark_ann_return = _annualized_return(buy_hold_equity)
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 and np.isfinite(ann_return) else np.nan
    benchmark_calmar = (
        benchmark_ann_return / abs(benchmark_max_drawdown)
        if benchmark_max_drawdown < 0 and np.isfinite(benchmark_ann_return)
        else np.nan
    )
    sharpe = float(strategy_return.mean() / strategy_return.std() * math.sqrt(252)) if strategy_return.std() > 0 else np.nan
    buy_hold_return = sim["buy_hold_return"].astype(float)
    benchmark_sharpe = (
        float(buy_hold_return.mean() / buy_hold_return.std() * math.sqrt(252))
        if buy_hold_return.std() > 0
        else np.nan
    )
    position_gap = (1.0 - position).clip(lower=0.0)
    missed_upside = float((position_gap * buy_hold_return.clip(lower=0.0)).sum())
    avoided_downside = float((position_gap * (-buy_hold_return.clip(upper=0.0))).sum())
    excess = total_return - benchmark_total if np.isfinite(total_return) and np.isfinite(benchmark_total) else np.nan
    drawdown_improvement = (
        max_drawdown - benchmark_max_drawdown
        if np.isfinite(max_drawdown) and np.isfinite(benchmark_max_drawdown)
        else np.nan
    )
    return {
        "total_return": total_return,
        "benchmark_total_return": benchmark_total,
        "excess_return_vs_buy_hold": excess,
        "max_drawdown": max_drawdown,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "drawdown_improvement": drawdown_improvement,
        "sharpe": sharpe,
        "benchmark_sharpe": benchmark_sharpe,
        "sharpe_improvement": sharpe - benchmark_sharpe if np.isfinite(sharpe) and np.isfinite(benchmark_sharpe) else np.nan,
        "calmar": calmar,
        "benchmark_calmar": benchmark_calmar,
        "calmar_improvement": calmar - benchmark_calmar if np.isfinite(calmar) and np.isfinite(benchmark_calmar) else np.nan,
        "avg_position": float(position.mean()) if len(position) else np.nan,
        "min_position": float(position.min()) if len(position) else np.nan,
        "avg_turnover": float(turnover.mean()) if len(turnover) else np.nan,
        "total_turnover": float(turnover.sum()),
        "transaction_cost": float(transaction_cost.sum()),
        "missed_upside": missed_upside,
        "avoided_downside": avoided_downside,
        "missed_minus_avoided": max(0.0, missed_upside - avoided_downside),
        "pct_days_below_full_exposure": float((position < 0.995).mean()) if len(position) else np.nan,
    }


def _rank_soft_confirmation_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return rows
    positive_cols = [
        "excess_return_vs_buy_hold",
        "drawdown_improvement",
        "calmar_improvement",
        "sharpe_improvement",
    ]
    for col in positive_cols:
        df[f"{col}_rank"] = df[col].rank(pct=True, na_option="bottom")
    df["missed_penalty_rank"] = df["missed_minus_avoided"].rank(pct=True, na_option="bottom")
    df["validation_score"] = (
        0.35 * df["excess_return_vs_buy_hold_rank"]
        + 0.25 * df["drawdown_improvement_rank"]
        + 0.25 * df["calmar_improvement_rank"]
        + 0.15 * df["sharpe_improvement_rank"]
        - 0.30 * df["missed_penalty_rank"]
    )
    ranked_records = df.sort_values("validation_score", ascending=False).to_dict("records")
    return ranked_records


def _tune_one_soft_confirmation_strategy(
    family: str,
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
    base_params: RiskBudgetParams | EarlyStressRiskBudgetParams,
    feature_set: str,
    lite_mode: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if lite_mode:
        grid = product(
            [1.00],
            [0.98, 0.99],
            [0.99, 1.00],
            [0.95, 0.96, 0.98],
            [0.92, 0.94],
            [0.92, 0.95],
            [0.98, 0.99, 1.00],
            [0.99, 1.00],
            [False, True],
        )
    else:
        grid = product(
            [0.98],
            [0.94, 0.95, 0.96],
            [0.96],
            [0.88, 0.90, 0.92],
            [0.85],
            [0.85, 0.90],
            [0.95, 0.98],
            [0.98],
            [False, True],
        )
    for (
        high_unconfirmed_weight,
        high_confirm_weight,
        extreme_unconfirmed_weight,
        extreme_confirm_weight,
        extreme_stress_weight,
        soft_min_weight,
        style_floor,
        strong_style_floor,
        require_persistence,
    ) in grid:
        if "risk_budget" in family and "early_stress" not in family:
            params = _soft_risk_param_from_base(
                base_params,  # type: ignore[arg-type]
                feature_set,
                high_confirm_weight,
                extreme_confirm_weight,
                soft_min_weight,
                style_floor,
                require_persistence,
                high_unconfirmed_weight=high_unconfirmed_weight,
                extreme_unconfirmed_weight=extreme_unconfirmed_weight,
                extreme_stress_weight=extreme_stress_weight,
                strong_style_floor=strong_style_floor,
            )
        else:
            params = _soft_early_param_from_base(
                base_params,  # type: ignore[arg-type]
                feature_set,
                high_confirm_weight,
                extreme_confirm_weight,
                soft_min_weight,
                style_floor,
                require_persistence,
                high_unconfirmed_weight=high_unconfirmed_weight,
                extreme_unconfirmed_weight=extreme_unconfirmed_weight,
                extreme_stress_weight=extreme_stress_weight,
                strong_style_floor=strong_style_floor,
            )
        frame = prepare_strategy_frame(feature_df, validation_preds, params)
        frame = frame.loc[
            (frame["date"] >= pd.Timestamp(VALID_START)) & (frame["date"] <= pd.Timestamp(VALID_END))
        ].copy().reset_index(drop=True)
        if len(frame) < 40:
            continue
        if isinstance(params, ExternalSoftConfirmRiskBudgetParams):
            position = (
                ml_external_soft_lite_risk_budget_position(frame, params)
                if lite_mode
                else ml_external_soft_confirm_risk_budget_position(frame, params)
            )
        else:
            position = (
                ml_external_soft_lite_early_stress_position(frame, params)
                if lite_mode
                else ml_external_soft_confirm_early_stress_position(frame, params)
            )
        metrics = _evaluate_soft_confirmation_frame(frame, position)
        rows.append(
            {
                "family": family,
                "feature_set": feature_set,
                "mode": "soft_lite" if lite_mode else "soft_confirm",
                "params": params.to_dict(),
                **metrics,
            }
        )

    ranked = _rank_soft_confirmation_candidates(rows)
    if not ranked:
        raise RuntimeError(f"No valid soft confirmation candidate was evaluated for {family}.")
    best = ranked[0]
    if "risk_budget" in family and "early_stress" not in family:
        selected_params = ExternalSoftConfirmRiskBudgetParams(**best["params"])
    else:
        selected_params = ExternalSoftConfirmEarlyStressParams(**best["params"])
    return {
        "family": family,
        "selected": best,
        "selected_params": selected_params,
        "candidate_count": len(rows),
        "top_candidates": ranked[:20],
        "validation_period": {
            "start": str(pd.Timestamp(VALID_START).date()),
            "end": str(pd.Timestamp(VALID_END).date()),
            "sample_count": int(
                len(
                    prepare_strategy_frame(feature_df, validation_preds, selected_params).loc[
                        lambda df: (df["date"] >= pd.Timestamp(VALID_START))
                        & (df["date"] <= pd.Timestamp(VALID_END))
                    ]
                )
            ),
        },
        "selection_rule": (
            "Soft external confirmation parameters are selected on validation only using rank-normalized score: "
            "0.35*excess rank + 0.25*drawdown-improvement rank + 0.25*calmar-improvement rank "
            "+ 0.15*sharpe-improvement rank - 0.30*missed-upside-minus-avoided-downside penalty rank."
            + (
                " Soft-lite mode uses higher external weights and a tighter external no-trade band, so it tests a higher-participation overlay."
                if lite_mode
                else ""
            )
        ),
    }


def tune_external_soft_confirmation_params(
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
    risk_budget_params: RiskBudgetParams,
    early_stress_params: EarlyStressRiskBudgetParams,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for feature_set, family in EXTERNAL_SOFT_RISK_BUDGET_STRATEGY_NAMES.items():
        out[family] = _tune_one_soft_confirmation_strategy(
            family,
            feature_df,
            validation_preds,
            risk_budget_params,
            feature_set,
        )
    for feature_set, family in EXTERNAL_SOFT_EARLY_STRESS_STRATEGY_NAMES.items():
        out[family] = _tune_one_soft_confirmation_strategy(
            family,
            feature_df,
            validation_preds,
            early_stress_params,
            feature_set,
        )
    for feature_set, family in EXTERNAL_SOFT_LITE_RISK_BUDGET_STRATEGY_NAMES.items():
        out[family] = _tune_one_soft_confirmation_strategy(
            family,
            feature_df,
            validation_preds,
            risk_budget_params,
            feature_set,
            lite_mode=True,
        )
    for feature_set, family in EXTERNAL_SOFT_LITE_EARLY_STRESS_STRATEGY_NAMES.items():
        out[family] = _tune_one_soft_confirmation_strategy(
            family,
            feature_df,
            validation_preds,
            early_stress_params,
            feature_set,
            lite_mode=True,
        )
    return out


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(write_json_ready(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def iter_upside_participation_param_candidates(
    selected_signal_model: dict[str, Any],
) -> list[UpsideParticipationParams]:
    selected = selected_signal_model["selected"]
    params: list[UpsideParticipationParams] = []
    grid = product(
        [0.55, 0.58, 0.60, 0.62],
        [0.40, 0.42, 0.45, 0.48],
        [1.00, 0.99, 0.98],
        [0.98, 0.97, 0.95],
        [0.95, 0.93, 0.90],
        [0.90, 0.88],
        [1.00, 0.99, 0.98],
        [1.15, 1.20, 1.25],
        [-0.08, -0.10, -0.12],
        [0.00, 0.03, 0.05],
    )
    for (
        high_signal_cut,
        low_signal_cut,
        neutral_position,
        mild_cut_position,
        risk_cut_position,
        min_position,
        strong_trend_floor,
        high_vol_multiplier,
        drawdown_cut,
        no_trade_band,
    ) in grid:
        if high_signal_cut <= low_signal_cut:
            continue
        if risk_cut_position > mild_cut_position:
            continue
        if min_position > risk_cut_position:
            continue
        params.append(
            UpsideParticipationParams(
                model_name=str(selected["model_name"]),
                horizon=int(selected["horizon"]),
                signal_name=str(selected["signal_name"]),
                high_signal_cut=high_signal_cut,
                low_signal_cut=low_signal_cut,
                neutral_position=neutral_position,
                mild_cut_position=mild_cut_position,
                risk_cut_position=risk_cut_position,
                min_position=min_position,
                strong_trend_floor=strong_trend_floor,
                high_vol_multiplier=high_vol_multiplier,
                drawdown_cut=drawdown_cut,
                no_trade_band=no_trade_band,
            )
        )
    return params


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def _annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    total = equity.iloc[-1] / equity.iloc[0]
    years = len(equity) / 252
    if years <= 0 or total <= 0:
        return np.nan
    return float(total ** (1 / years) - 1)


def _evaluate_cost_aware_frame(frame: pd.DataFrame, final_position: pd.Series) -> dict[str, float]:
    sim = simulate_strategy_cost_aware(
        frame.reset_index(drop=True),
        final_position.reset_index(drop=True),
        "validation_candidate",
        initial_position=1.0,
    )
    position = sim["final_position"].astype(float)
    turnover = sim["turnover"].astype(float)
    transaction_cost = sim["transaction_cost"].astype(float)
    strategy_return = sim["strategy_return"].astype(float)
    equity = sim["equity"].astype(float)
    buy_hold_equity = sim["buy_hold_equity"].astype(float)
    drawdown = sim["drawdown"].astype(float)
    buy_hold_drawdown = sim["buy_hold_drawdown"].astype(float)
    total_return = float(equity.iloc[-1] / INITIAL_CAPITAL - 1) if len(equity) else np.nan
    benchmark_total = float(buy_hold_equity.iloc[-1] / INITIAL_CAPITAL - 1) if len(buy_hold_equity) else np.nan
    max_drawdown = float(drawdown.min()) if len(drawdown) else np.nan
    benchmark_max_drawdown = float(buy_hold_drawdown.min()) if len(buy_hold_drawdown) else np.nan
    sharpe = float(strategy_return.mean() / strategy_return.std() * math.sqrt(252)) if strategy_return.std() > 0 else np.nan
    excess = total_return - benchmark_total if np.isfinite(total_return) and np.isfinite(benchmark_total) else np.nan
    drawdown_improvement = (
        abs(benchmark_max_drawdown) - abs(max_drawdown)
        if np.isfinite(benchmark_max_drawdown) and np.isfinite(max_drawdown)
        else np.nan
    )
    avg_position = float(position.mean()) if len(position) else np.nan
    avg_turnover = float(turnover.mean()) if len(turnover) else np.nan
    excess_drawdown_pain = (
        max(0.0, abs(max_drawdown) - abs(benchmark_max_drawdown))
        if np.isfinite(max_drawdown) and np.isfinite(benchmark_max_drawdown)
        else 0.0
    )
    score = (
        2.00 * (excess if np.isfinite(excess) else 0.0)
        + 0.60 * (drawdown_improvement if np.isfinite(drawdown_improvement) else 0.0)
        + 0.15 * (sharpe if np.isfinite(sharpe) else 0.0)
        - 0.10 * (avg_turnover if np.isfinite(avg_turnover) else 0.0)
        - 0.30 * excess_drawdown_pain
        - 0.20 * max(0.0, 0.97 - (avg_position if np.isfinite(avg_position) else 0.0))
    )
    return {
        "total_return": total_return,
        "benchmark_total_return": benchmark_total,
        "excess_return_vs_buy_hold": excess,
        "max_drawdown": max_drawdown,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "drawdown_improvement": drawdown_improvement,
        "sharpe": sharpe,
        "avg_position": avg_position,
        "avg_turnover": avg_turnover,
        "total_turnover": float(turnover.sum()),
        "transaction_cost": float(transaction_cost.sum()),
        "pct_days_below_full_exposure": float((position < 0.995).mean()) if len(position) else np.nan,
        "score": float(score),
    }


def _evaluate_stability_aware_frame(frame: pd.DataFrame, final_position: pd.Series) -> dict[str, float]:
    row = _evaluate_cost_aware_frame(frame, final_position)
    max_drawdown = row["max_drawdown"]
    benchmark_max_drawdown = row["benchmark_max_drawdown"]
    excess_drawdown_pain = (
        max(0.0, abs(max_drawdown) - abs(benchmark_max_drawdown))
        if np.isfinite(max_drawdown) and np.isfinite(benchmark_max_drawdown)
        else 0.0
    )
    row["score"] = float(
        1.80 * (row["excess_return_vs_buy_hold"] if np.isfinite(row["excess_return_vs_buy_hold"]) else 0.0)
        + 0.70 * (row["drawdown_improvement"] if np.isfinite(row["drawdown_improvement"]) else 0.0)
        + 0.10 * (row["sharpe"] if np.isfinite(row["sharpe"]) else 0.0)
        - 0.12 * (row["avg_turnover"] if np.isfinite(row["avg_turnover"]) else 0.0)
        - 0.40 * excess_drawdown_pain
        - 0.20 * max(0.0, 0.985 - (row["avg_position"] if np.isfinite(row["avg_position"]) else 0.0))
        - 0.10 * max(0.0, (row["pct_days_below_full_exposure"] if np.isfinite(row["pct_days_below_full_exposure"]) else 0.0) - 0.35)
    )
    return row


def _validation_windows(validation_preds: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    windows = [
        ("validation_config", pd.Timestamp(VALID_START), pd.Timestamp(VALID_END)),
        ("rolling_2023h2", pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-31")),
        ("rolling_2024h1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")),
        ("rolling_2024h2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
    ]
    dates = pd.to_datetime(validation_preds["date"], errors="coerce")
    out: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    seen: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
    for name, start, end in windows:
        if (start, end) in seen:
            continue
        count = int(((dates >= start) & (dates <= end)).sum())
        if count >= 40:
            out.append((name, start, end))
            seen.add((start, end))
    if not out and dates.notna().any():
        out.append(("validation_available", dates.min(), dates.max()))
    return out


def tune_upside_participation_enhancement_params(
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
    selected_signal_model: dict[str, Any],
) -> dict[str, Any]:
    candidates = iter_upside_participation_param_candidates(selected_signal_model)
    if not candidates:
        raise RuntimeError("No upside participation parameter candidates generated.")

    dummy = candidates[0]
    base_frame = prepare_strategy_frame(feature_df, validation_preds, dummy)
    windows = _validation_windows(validation_preds)
    window_frames: list[tuple[str, pd.DataFrame]] = []
    for name, start, end in windows:
        frame = base_frame.loc[(base_frame["date"] >= start) & (base_frame["date"] <= end)].copy().reset_index(drop=True)
        if len(frame) >= 40:
            window_frames.append((name, frame))
    if not window_frames:
        raise RuntimeError("No validation windows have enough samples for upside participation tuning.")

    best: dict[str, Any] | None = None
    top: list[dict[str, Any]] = []
    evaluated_count = 0

    for params in candidates:
        scores: list[float] = []
        metrics_by_window: list[dict[str, Any]] = []
        for window_name, frame in window_frames:
            position = ml_upside_participation_enhancement_position(frame, params)
            row = _evaluate_cost_aware_frame(frame, position)
            row["window"] = window_name
            scores.append(float(row["score"]))
            metrics_by_window.append(row)
        if not scores:
            continue
        evaluated_count += 1
        mean_score = float(np.mean(scores))
        min_score = float(np.min(scores))
        std_score = float(np.std(scores))
        robust_score = mean_score + 0.50 * min_score - 0.20 * std_score
        validation_mean_excess = float(np.mean([m["excess_return_vs_buy_hold"] for m in metrics_by_window]))
        validation_min_excess = float(np.min([m["excess_return_vs_buy_hold"] for m in metrics_by_window]))
        validation_mean_position = float(np.mean([m["avg_position"] for m in metrics_by_window]))
        candidate = {
            "family": UPSIDE_STRATEGY_NAME,
            "robust_score": robust_score,
            "mean_score": mean_score,
            "min_score": min_score,
            "std_score": std_score,
            "validation_mean_excess_return": validation_mean_excess,
            "validation_min_excess_return": validation_min_excess,
            "validation_mean_position": validation_mean_position,
            "params": params.to_dict(),
            "metrics_by_window": metrics_by_window,
        }
        if best is None or robust_score > best["robust_score"]:
            best = candidate
        top.append(candidate)

    if best is None:
        raise RuntimeError("No valid upside participation parameter candidate was evaluated.")

    ranked = sorted(top, key=lambda item: item["robust_score"], reverse=True)
    selected_params = UpsideParticipationParams(**best["params"])
    return {
        "family": UPSIDE_STRATEGY_NAME,
        "selected": best,
        "selected_params": selected_params,
        "candidate_count": evaluated_count,
        "top_candidates": ranked[:20],
        "validation_windows_used": [
            {
                "name": name,
                "start": str(frame["date"].min().date()),
                "end": str(frame["date"].max().date()),
                "sample_count": int(len(frame)),
            }
            for name, frame in window_frames
        ],
        "selection_rule": (
            "Parameters are selected using validation windows only. Robust score = mean(score_by_window) + "
            "0.50*min(score_by_window) - 0.20*std(score_by_window). Single-window score rewards positive excess "
            "and drawdown improvement while penalizing turnover, low average position, and worse drawdown."
        ),
        "robustness_statistics": {
            "selected_robust_score": best["robust_score"],
            "selected_mean_score": best["mean_score"],
            "selected_min_score": best["min_score"],
            "selected_std_score": best["std_score"],
            "selected_validation_mean_excess_return": best["validation_mean_excess_return"],
            "selected_validation_min_excess_return": best["validation_min_excess_return"],
            "selected_validation_mean_position": best["validation_mean_position"],
        },
    }


def _primary_component(selected_components: list[dict[str, Any]]) -> dict[str, Any]:
    if not selected_components:
        raise ValueError("No stability-aware selected components were provided.")
    return selected_components[0]


def iter_stability_aware_high_participation_param_candidates(
    selected_components: list[dict[str, Any]],
) -> list[StabilityAwareHighParticipationParams]:
    primary = _primary_component(selected_components)
    params: list[StabilityAwareHighParticipationParams] = []
    grid = product(
        [0.15, 0.20, 0.25],
        [0.05, 0.10, 0.15],
        [0.99, 0.98, 0.97],
        [0.97, 0.95, 0.94],
        [0.95, 0.94, 0.92],
        [1.15, 1.20, 1.25],
        [-0.08, -0.10, -0.12],
        [0.00, 0.03, 0.05],
        [True, False],
    )
    for (
        low_score_cut,
        very_low_score_cut,
        mild_cut_position,
        risk_cut_position,
        min_position,
        high_vol_multiplier,
        drawdown_cut,
        no_trade_band,
        require_risk_confirm,
    ) in grid:
        if very_low_score_cut > low_score_cut:
            continue
        if risk_cut_position > mild_cut_position:
            continue
        if min_position > risk_cut_position:
            continue
        params.append(
            StabilityAwareHighParticipationParams(
                model_name=str(primary["model_name"]),
                horizon=int(primary["horizon"]),
                low_score_cut=low_score_cut,
                very_low_score_cut=very_low_score_cut,
                mild_cut_position=mild_cut_position,
                risk_cut_position=risk_cut_position,
                min_position=min_position,
                high_vol_multiplier=high_vol_multiplier,
                drawdown_cut=drawdown_cut,
                no_trade_band=no_trade_band,
                require_risk_confirm=bool(require_risk_confirm),
            )
        )
    return params


def _prepare_stability_aware_frame(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    selected_components: list[dict[str, Any]],
    score_df: pd.DataFrame,
) -> pd.DataFrame:
    primary = _primary_component(selected_components)
    dummy = StabilityAwareHighParticipationParams(
        model_name=str(primary["model_name"]),
        horizon=int(primary["horizon"]),
        low_score_cut=0.20,
        very_low_score_cut=0.10,
        mild_cut_position=0.99,
        risk_cut_position=0.97,
        min_position=0.95,
        high_vol_multiplier=1.20,
        drawdown_cut=-0.10,
        no_trade_band=0.03,
        require_risk_confirm=True,
    )
    pred_with_score = predictions.merge(score_df, on="date", how="left")
    frame = prepare_strategy_frame(feature_df, pred_with_score, dummy)
    frame["ensemble_upside_score"] = frame["ensemble_upside_score"].fillna(0.5)
    return frame


def tune_stability_aware_high_participation_params(
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
    selected_components: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = iter_stability_aware_high_participation_param_candidates(selected_components)
    if not candidates:
        raise RuntimeError("No stability-aware high-participation parameter candidates generated.")
    validation_score = build_stability_aware_upside_score(validation_preds, selected_components)
    base_frame = _prepare_stability_aware_frame(feature_df, validation_preds, selected_components, validation_score)
    base_frame = base_frame.loc[
        (base_frame["date"] >= pd.Timestamp(VALID_START)) & (base_frame["date"] <= pd.Timestamp(VALID_END))
    ].copy().reset_index(drop=True)
    if len(base_frame) < 40:
        raise RuntimeError("Validation period does not have enough rows for stability-aware high-participation tuning.")

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for params in candidates:
        position = ml_stability_aware_high_participation_position(base_frame, params)
        metrics = _evaluate_stability_aware_frame(base_frame, position)
        candidate = {
            "family": STABILITY_AWARE_STRATEGY_NAME,
            "validation_score": metrics["score"],
            "params": params.to_dict(),
            **metrics,
        }
        rows.append(candidate)
        if best is None or candidate["validation_score"] > best["validation_score"]:
            best = candidate
    if best is None:
        raise RuntimeError("No valid stability-aware high-participation candidate was evaluated.")

    ranked = sorted(rows, key=lambda item: item["validation_score"], reverse=True)
    selected_params = StabilityAwareHighParticipationParams(**best["params"])
    return {
        "family": STABILITY_AWARE_STRATEGY_NAME,
        "selected": best,
        "selected_params": selected_params,
        "candidate_count": len(rows),
        "top_candidates": ranked[:20],
        "selected_signal_components": selected_components,
        "validation_period": {
            "start": str(pd.Timestamp(VALID_START).date()),
            "end": str(pd.Timestamp(VALID_END).date()),
            "sample_count": int(len(base_frame)),
        },
        "selection_rule": (
            "Parameters are selected only on the configured validation period. Score = "
            "1.80*excess_return_vs_buy_hold + 0.70*drawdown_improvement + 0.10*sharpe "
            "- 0.12*avg_turnover - 0.40*worse_drawdown_penalty - 0.20*low_avg_position_penalty "
            "- 0.10*excess_days_below_full_penalty. This is high-participation index enhancement, not long/cash timing."
        ),
    }


def _risk_budget_param_from_core(
    core: dict[str, Any],
    model_name: str,
    horizon: int,
    positive_prob: float,
    big_up_prob_cut: float,
    recovery_position: float,
) -> RiskBudgetParams:
    return RiskBudgetParams(
        model_name=str(model_name),
        horizon=int(horizon),
        high_vol_multiplier=float(core["high_vol_multiplier"]),
        drawdown_cut=float(core["drawdown_cut"]),
        mild_risk_cut=float(core["mild_risk_cut"]),
        high_risk_cut=float(core["high_risk_cut"]),
        severe_risk_cut=float(core["severe_risk_cut"]),
        mild_position=float(core["mild_position"]),
        high_risk_position=float(core["high_risk_position"]),
        severe_position=float(core["severe_position"]),
        min_position=float(core["min_position"]),
        positive_prob=float(positive_prob),
        big_up_prob_cut=float(big_up_prob_cut),
        recovery_position=float(recovery_position),
        no_trade_band=float(core["no_trade_band"]),
    )


def _iter_risk_budget_core_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid = product(
        [1.10, 1.15, 1.20, 1.25],
        [-0.06, -0.08, -0.10, -0.12],
        [0.25, 0.30, 0.35],
        [0.45, 0.50, 0.55],
        [0.65, 0.70],
        [0.95, 0.92, 0.90],
        [0.90, 0.85, 0.80],
        [0.80, 0.75, 0.70],
        [0.70, 0.75, 0.80],
        [0.00, 0.03, 0.05],
    )
    for (
        high_vol_multiplier,
        drawdown_cut,
        mild_risk_cut,
        high_risk_cut,
        severe_risk_cut,
        mild_position,
        high_risk_position,
        severe_position,
        min_position,
        no_trade_band,
    ) in grid:
        if not (mild_risk_cut < high_risk_cut < severe_risk_cut):
            continue
        if high_risk_position > mild_position:
            continue
        if severe_position > high_risk_position:
            continue
        if min_position > severe_position:
            continue
        rows.append(
            {
                "high_vol_multiplier": high_vol_multiplier,
                "drawdown_cut": drawdown_cut,
                "mild_risk_cut": mild_risk_cut,
                "high_risk_cut": high_risk_cut,
                "severe_risk_cut": severe_risk_cut,
                "mild_position": mild_position,
                "high_risk_position": high_risk_position,
                "severe_position": severe_position,
                "min_position": min_position,
                "no_trade_band": no_trade_band,
            }
        )
    return rows


def _evaluate_risk_budget_frame(frame: pd.DataFrame, final_position: pd.Series) -> dict[str, float]:
    row = _evaluate_cost_aware_frame(frame, final_position)
    max_drawdown = row["max_drawdown"]
    benchmark_max_drawdown = row["benchmark_max_drawdown"]
    drawdown_improvement = (
        max_drawdown - benchmark_max_drawdown
        if np.isfinite(max_drawdown) and np.isfinite(benchmark_max_drawdown)
        else np.nan
    )
    worse_drawdown_penalty = (
        max(0.0, abs(max_drawdown) - abs(benchmark_max_drawdown))
        if np.isfinite(max_drawdown) and np.isfinite(benchmark_max_drawdown)
        else 0.0
    )
    row["drawdown_improvement"] = drawdown_improvement
    row["score"] = float(
        1.20 * (row["excess_return_vs_buy_hold"] if np.isfinite(row["excess_return_vs_buy_hold"]) else 0.0)
        + 1.00 * (drawdown_improvement if np.isfinite(drawdown_improvement) else 0.0)
        + 0.20 * (row["sharpe"] if np.isfinite(row["sharpe"]) else 0.0)
        - 0.10 * (row["avg_turnover"] if np.isfinite(row["avg_turnover"]) else 0.0)
        - 0.20 * max(0.0, 0.92 - (row["avg_position"] if np.isfinite(row["avg_position"]) else 0.0))
        - 0.30 * worse_drawdown_penalty
    )
    return row


def tune_risk_budget_enhancement_params(
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
) -> dict[str, Any]:
    valid = validation_preds.loc[
        (pd.to_datetime(validation_preds["date"]) >= pd.Timestamp(VALID_START))
        & (pd.to_datetime(validation_preds["date"]) <= pd.Timestamp(VALID_END))
    ].copy()
    model_pairs = valid[["model_name", "horizon"]].drop_duplicates().sort_values(["model_name", "horizon"])
    if model_pairs.empty:
        raise RuntimeError("No validation predictions available for risk-budget tuning.")

    first_pair = model_pairs.iloc[0]
    core_candidates = _iter_risk_budget_core_candidates()
    if not core_candidates:
        raise RuntimeError("No risk-budget core candidates generated.")

    dummy_core = core_candidates[0]
    dummy = _risk_budget_param_from_core(
        dummy_core,
        str(first_pair["model_name"]),
        int(first_pair["horizon"]),
        positive_prob=2.0,
        big_up_prob_cut=2.0,
        recovery_position=1.0,
    )
    core_frame = prepare_strategy_frame(feature_df, validation_preds, dummy)
    core_frame = core_frame.loc[
        (core_frame["date"] >= pd.Timestamp(VALID_START)) & (core_frame["date"] <= pd.Timestamp(VALID_END))
    ].copy().reset_index(drop=True)
    if len(core_frame) < 40:
        raise RuntimeError("Validation period does not have enough rows for risk-budget tuning.")

    core_rows: list[dict[str, Any]] = []
    for core in core_candidates:
        params = _risk_budget_param_from_core(
            core,
            str(first_pair["model_name"]),
            int(first_pair["horizon"]),
            positive_prob=2.0,
            big_up_prob_cut=2.0,
            recovery_position=1.0,
        )
        position = ml_risk_budget_enhancement_position(core_frame, params)
        metrics = _evaluate_risk_budget_frame(core_frame, position)
        core_rows.append({"risk_core": core, "risk_core_score": metrics["score"], **metrics})

    core_rows = sorted(core_rows, key=lambda item: item["risk_core_score"], reverse=True)
    shortlisted_cores = core_rows[:80]
    if not shortlisted_cores:
        raise RuntimeError("No risk-budget core candidates survived validation scoring.")

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    ml_grid = list(product([0.50, 0.52, 0.55], [0.50, 0.55, 0.60], [0.95, 0.98, 1.00]))

    for pair in model_pairs.itertuples(index=False):
        model_name = str(pair.model_name)
        horizon = int(pair.horizon)
        frame_dummy = _risk_budget_param_from_core(
            shortlisted_cores[0]["risk_core"],
            model_name,
            horizon,
            positive_prob=0.50,
            big_up_prob_cut=0.50,
            recovery_position=1.0,
        )
        frame = prepare_strategy_frame(feature_df, validation_preds, frame_dummy)
        frame = frame.loc[
            (frame["date"] >= pd.Timestamp(VALID_START)) & (frame["date"] <= pd.Timestamp(VALID_END))
        ].copy().reset_index(drop=True)
        if len(frame) < 40:
            continue
        for core_row in shortlisted_cores:
            core = core_row["risk_core"]
            for positive_prob, big_up_prob_cut, recovery_position in ml_grid:
                params = _risk_budget_param_from_core(
                    core,
                    model_name,
                    horizon,
                    positive_prob=positive_prob,
                    big_up_prob_cut=big_up_prob_cut,
                    recovery_position=recovery_position,
                )
                position = ml_risk_budget_enhancement_position(frame, params)
                metrics = _evaluate_risk_budget_frame(frame, position)
                candidate = {
                    "family": RISK_BUDGET_STRATEGY_NAME,
                    "validation_score": metrics["score"],
                    "params": params.to_dict(),
                    **metrics,
                }
                rows.append(candidate)
                if best is None or candidate["validation_score"] > best["validation_score"]:
                    best = candidate

    if best is None:
        raise RuntimeError("No valid risk-budget enhancement candidate was evaluated.")

    ranked = sorted(rows, key=lambda item: item["validation_score"], reverse=True)
    selected_params = RiskBudgetParams(**best["params"])
    return {
        "family": RISK_BUDGET_STRATEGY_NAME,
        "selected": best,
        "selected_params": selected_params,
        "candidate_count": len(rows),
        "risk_core_candidate_count": len(core_candidates),
        "risk_core_shortlist_count": len(shortlisted_cores),
        "top_candidates": ranked[:20],
        "top_risk_core_candidates": core_rows[:20],
        "validation_period": {
            "start": str(pd.Timestamp(VALID_START).date()),
            "end": str(pd.Timestamp(VALID_END).date()),
            "sample_count": int(len(core_frame)),
        },
        "selection_rule": (
            "Parameters are selected only from the configured validation period. The search first scores risk-budget "
            "core exposure rules, then evaluates all validation model/horizon pairs with ML recovery thresholds on "
            "the best risk-core shortlist. Score = 1.20*excess_return_vs_buy_hold + 1.00*drawdown_improvement + "
            "0.20*sharpe - 0.10*avg_turnover - 0.20*low_avg_position_penalty - 0.30*worse_drawdown_penalty."
        ),
    }


def _early_stress_param_from_core(
    core: dict[str, Any],
    model_name: str,
    horizon: int,
    weak_prob: float,
    strong_up_prob: float,
    big_up_prob_cut: float,
) -> EarlyStressRiskBudgetParams:
    return EarlyStressRiskBudgetParams(
        model_name=str(model_name),
        horizon=int(horizon),
        early_stress_cut=float(core["early_stress_cut"]),
        severe_stress_cut=float(core["severe_stress_cut"]),
        weak_prob=float(weak_prob),
        strong_up_prob=float(strong_up_prob),
        big_up_prob_cut=float(big_up_prob_cut),
        mild_position=float(core["mild_position"]),
        stress_position=float(core["stress_position"]),
        severe_position=float(core["severe_position"]),
        min_position=float(core["min_position"]),
        no_trade_band=float(core["no_trade_band"]),
        require_ml_not_positive=bool(core["require_ml_not_positive"]),
        recovery_fast=bool(core["recovery_fast"]),
    )


def _iter_early_stress_core_candidates() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid = product(
        [0.45, 0.50, 0.55, 0.60],
        [0.65, 0.70, 0.75],
        [0.98, 0.95],
        [0.92, 0.90, 0.88],
        [0.85, 0.80],
        [0.80, 0.85, 0.88],
        [0.00, 0.03, 0.05],
        [True, False],
        [True, False],
    )
    for (
        early_stress_cut,
        severe_stress_cut,
        mild_position,
        stress_position,
        severe_position,
        min_position,
        no_trade_band,
        require_ml_not_positive,
        recovery_fast,
    ) in grid:
        if severe_stress_cut <= early_stress_cut:
            continue
        if stress_position > mild_position:
            continue
        if severe_position > stress_position:
            continue
        if min_position > severe_position:
            continue
        rows.append(
            {
                "early_stress_cut": early_stress_cut,
                "severe_stress_cut": severe_stress_cut,
                "mild_position": mild_position,
                "stress_position": stress_position,
                "severe_position": severe_position,
                "min_position": min_position,
                "no_trade_band": no_trade_band,
                "require_ml_not_positive": require_ml_not_positive,
                "recovery_fast": recovery_fast,
            }
        )
    return rows


def _evaluate_early_stress_frame(frame: pd.DataFrame, final_position: pd.Series) -> dict[str, float]:
    row = _evaluate_cost_aware_frame(frame, final_position)
    max_drawdown = row["max_drawdown"]
    benchmark_max_drawdown = row["benchmark_max_drawdown"]
    drawdown_improvement = (
        max_drawdown - benchmark_max_drawdown
        if np.isfinite(max_drawdown) and np.isfinite(benchmark_max_drawdown)
        else np.nan
    )
    worse_drawdown_penalty = (
        max(0.0, abs(max_drawdown) - abs(benchmark_max_drawdown))
        if np.isfinite(max_drawdown) and np.isfinite(benchmark_max_drawdown)
        else 0.0
    )
    row["drawdown_improvement"] = drawdown_improvement
    row["score"] = float(
        1.50 * (row["excess_return_vs_buy_hold"] if np.isfinite(row["excess_return_vs_buy_hold"]) else 0.0)
        + 1.00 * (drawdown_improvement if np.isfinite(drawdown_improvement) else 0.0)
        + 0.15 * (row["sharpe"] if np.isfinite(row["sharpe"]) else 0.0)
        - 0.10 * (row["avg_turnover"] if np.isfinite(row["avg_turnover"]) else 0.0)
        - 0.20 * max(0.0, 0.97 - (row["avg_position"] if np.isfinite(row["avg_position"]) else 0.0))
        - 0.30 * worse_drawdown_penalty
    )
    return row


def tune_early_stress_risk_budget_params(
    feature_df: pd.DataFrame,
    validation_preds: pd.DataFrame,
) -> dict[str, Any]:
    valid = validation_preds.loc[
        (pd.to_datetime(validation_preds["date"]) >= pd.Timestamp(VALID_START))
        & (pd.to_datetime(validation_preds["date"]) <= pd.Timestamp(VALID_END))
    ].copy()
    model_pairs = valid[["model_name", "horizon"]].drop_duplicates().sort_values(["model_name", "horizon"])
    if model_pairs.empty:
        raise RuntimeError("No validation predictions available for early-stress risk-budget tuning.")

    first_pair = model_pairs.iloc[0]
    core_candidates = _iter_early_stress_core_candidates()
    if not core_candidates:
        raise RuntimeError("No early-stress risk core candidates generated.")

    dummy = _early_stress_param_from_core(
        core_candidates[0],
        str(first_pair["model_name"]),
        int(first_pair["horizon"]),
        weak_prob=0.45,
        strong_up_prob=2.0,
        big_up_prob_cut=2.0,
    )
    core_frame = prepare_strategy_frame(feature_df, validation_preds, dummy)
    core_frame = core_frame.loc[
        (core_frame["date"] >= pd.Timestamp(VALID_START)) & (core_frame["date"] <= pd.Timestamp(VALID_END))
    ].copy().reset_index(drop=True)
    if len(core_frame) < 40:
        raise RuntimeError("Validation period does not have enough rows for early-stress risk-budget tuning.")

    core_rows: list[dict[str, Any]] = []
    for core in core_candidates:
        params = _early_stress_param_from_core(
            core,
            str(first_pair["model_name"]),
            int(first_pair["horizon"]),
            weak_prob=0.45,
            strong_up_prob=2.0,
            big_up_prob_cut=2.0,
        )
        position = ml_early_stress_risk_budget_position(core_frame, params)
        metrics = _evaluate_early_stress_frame(core_frame, position)
        core_rows.append({"risk_core": core, "risk_core_score": metrics["score"], **metrics})
    core_rows = sorted(core_rows, key=lambda item: item["risk_core_score"], reverse=True)
    shortlisted_cores = core_rows[:100]
    if not shortlisted_cores:
        raise RuntimeError("No early-stress risk core candidates survived validation scoring.")

    ml_grid = list(product([0.45, 0.48, 0.50], [0.52, 0.55, 0.58], [0.52, 0.55, 0.58]))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for pair in model_pairs.itertuples(index=False):
        model_name = str(pair.model_name)
        horizon = int(pair.horizon)
        frame_dummy = _early_stress_param_from_core(
            shortlisted_cores[0]["risk_core"],
            model_name,
            horizon,
            weak_prob=0.45,
            strong_up_prob=0.52,
            big_up_prob_cut=0.52,
        )
        frame = prepare_strategy_frame(feature_df, validation_preds, frame_dummy)
        frame = frame.loc[
            (frame["date"] >= pd.Timestamp(VALID_START)) & (frame["date"] <= pd.Timestamp(VALID_END))
        ].copy().reset_index(drop=True)
        if len(frame) < 40:
            continue
        for core_row in shortlisted_cores:
            core = core_row["risk_core"]
            for weak_prob, strong_up_prob, big_up_prob_cut in ml_grid:
                params = _early_stress_param_from_core(
                    core,
                    model_name,
                    horizon,
                    weak_prob=weak_prob,
                    strong_up_prob=strong_up_prob,
                    big_up_prob_cut=big_up_prob_cut,
                )
                position = ml_early_stress_risk_budget_position(frame, params)
                metrics = _evaluate_early_stress_frame(frame, position)
                candidate = {
                    "family": EARLY_STRESS_STRATEGY_NAME,
                    "validation_score": metrics["score"],
                    "params": params.to_dict(),
                    **metrics,
                }
                rows.append(candidate)
                if best is None or candidate["validation_score"] > best["validation_score"]:
                    best = candidate

    if best is None:
        raise RuntimeError("No valid early-stress risk-budget candidate was evaluated.")

    ranked = sorted(rows, key=lambda item: item["validation_score"], reverse=True)
    selected_params = EarlyStressRiskBudgetParams(**best["params"])
    return {
        "family": EARLY_STRESS_STRATEGY_NAME,
        "selected": best,
        "selected_params": selected_params,
        "candidate_count": len(rows),
        "risk_core_candidate_count": len(core_candidates),
        "risk_core_shortlist_count": len(shortlisted_cores),
        "top_candidates": ranked[:20],
        "top_risk_core_candidates": core_rows[:20],
        "validation_period": {
            "start": str(pd.Timestamp(VALID_START).date()),
            "end": str(pd.Timestamp(VALID_END).date()),
            "sample_count": int(len(core_frame)),
        },
        "selection_rule": (
            "Parameters are selected only from the configured validation period. The search first scores early-stress "
            "core rules, then evaluates all validation model/horizon pairs with ML/upside thresholds on the best "
            "risk-core shortlist. Score = 1.50*excess_return_vs_buy_hold + 1.00*drawdown_improvement + 0.15*sharpe "
            "- 0.10*avg_turnover - 0.20*low_avg_position_penalty - 0.30*worse_drawdown_penalty."
        ),
    }


def build_upside_participation_attribution(
    strategy_daily: pd.DataFrame,
    selected_signal_name: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = strategy_daily.loc[strategy_daily["strategy"] == UPSIDE_STRATEGY_NAME].sort_values("date").copy()
    if df.empty:
        raise ValueError(f"No strategy rows for {UPSIDE_STRATEGY_NAME}")
    position_gap = (1.0 - df["final_position"].astype(float)).clip(lower=0.0)
    ret = df["buy_hold_return"].astype(float)
    df["position_gap_from_full"] = position_gap
    df["position_timing_contribution"] = position_gap * (-ret)
    missed_upside = float((position_gap * ret.clip(lower=0.0)).sum())
    avoided_downside = float((position_gap * (-ret.clip(upper=0.0))).sum())
    transaction_cost = float(df["transaction_cost"].sum())
    summary = {
        "days_below_full_exposure": int((df["final_position"] < 0.995).sum()),
        "avg_position": float(df["final_position"].mean()),
        "min_position": float(df["final_position"].min()),
        "max_position": float(df["final_position"].max()),
        "total_turnover": float(df["turnover"].sum()),
        "missed_upside": missed_upside,
        "avoided_downside": avoided_downside,
        "transaction_cost": transaction_cost,
        "net_position_timing_contribution": avoided_downside - missed_upside - transaction_cost,
    }
    if selected_signal_name in df.columns:
        df["selected_signal"] = df[selected_signal_name]
    else:
        df["selected_signal"] = np.nan
    return df, summary


def write_upside_participation_attribution_summary(
    strategy_daily: pd.DataFrame,
    selected_signal_name: str,
    path: Path,
) -> None:
    attribution, summary = build_upside_participation_attribution(strategy_daily, selected_signal_name)
    defensive = attribution.loc[attribution["final_position"] < 0.995].copy()
    best_defensive = defensive.loc[defensive["buy_hold_return"] < 0].sort_values(
        "position_timing_contribution",
        ascending=False,
    ).head(10)
    worst_defensive = defensive.loc[defensive["buy_hold_return"] > 0].sort_values(
        "position_timing_contribution",
    ).head(10)
    full_participation = attribution.loc[attribution["final_position"] >= 0.999].copy()
    high_upside_full = full_participation.sort_values(
        ["selected_signal", "buy_hold_return"],
        ascending=[False, False],
    ).head(10)

    def table(df: pd.DataFrame) -> str:
        cols = [
            "date",
            "final_position",
            "buy_hold_return",
            "position_timing_contribution",
            "selected_signal",
            "pred_ret",
        ]
        use_cols = [col for col in cols if col in df.columns]
        if df.empty or not use_cols:
            return "No rows."
        use = df.loc[:, use_cols]
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

    success = summary["avoided_downside"] > summary["missed_upside"] + summary["transaction_cost"]
    conclusion = (
        "The strategy successfully avoided unfavorable regimes."
        if success
        else "The strategy underperformance mainly comes from missed upside or weak downside identification."
    )
    text = f"""# Upside Participation Attribution Summary

## Summary
- Total days below full exposure: {int(summary["days_below_full_exposure"])}
- Average position: {_fmt(summary["avg_position"])}
- Minimum position: {_fmt(summary["min_position"])}
- Maximum position: {_fmt(summary["max_position"])}
- Total turnover: {_fmt(summary["total_turnover"])}
- Missed upside from reduced exposure on positive-return days: {_fmt(summary["missed_upside"])}
- Avoided downside from reduced exposure on negative-return days: {_fmt(summary["avoided_downside"])}
- Transaction cost: {_fmt(summary["transaction_cost"])}
- Net position timing contribution: {_fmt(summary["net_position_timing_contribution"])}

## Conclusion
{conclusion}

## Top 10 Best Defensive Days
{table(best_defensive)}

## Top 10 Worst Defensive Days
{table(worst_defensive)}

## Top 10 High-Upside Full Participation Days
{table(high_upside_full)}
"""
    path.write_text(text, encoding="utf-8")


def build_stability_aware_high_participation_attribution(
    strategy_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = strategy_daily.loc[strategy_daily["strategy"] == STABILITY_AWARE_STRATEGY_NAME].sort_values("date").copy()
    if df.empty:
        raise ValueError(f"No strategy rows for {STABILITY_AWARE_STRATEGY_NAME}")
    position_gap = (1.0 - df["final_position"].astype(float)).clip(lower=0.0)
    ret = df["buy_hold_return"].astype(float)
    df["position_gap_from_full"] = position_gap
    df["position_timing_contribution"] = position_gap * (-ret)
    missed_upside = float((position_gap * ret.clip(lower=0.0)).sum())
    avoided_downside = float((position_gap * (-ret.clip(upper=0.0))).sum())
    transaction_cost = float(df["transaction_cost"].sum())
    summary = {
        "days_below_full_exposure": int((df["final_position"] < 0.995).sum()),
        "pct_days_below_full_exposure": float((df["final_position"] < 0.995).mean()),
        "avg_position": float(df["final_position"].mean()),
        "min_position": float(df["final_position"].min()),
        "max_position": float(df["final_position"].max()),
        "total_turnover": float(df["turnover"].sum()),
        "missed_upside": missed_upside,
        "avoided_downside": avoided_downside,
        "transaction_cost": transaction_cost,
        "net_position_timing_contribution": avoided_downside - missed_upside - transaction_cost,
    }
    return df, summary


def write_stability_aware_high_participation_attribution_summary(
    strategy_daily: pd.DataFrame,
    selected_components: list[dict[str, Any]],
    path: Path,
) -> None:
    attribution, summary = build_stability_aware_high_participation_attribution(strategy_daily)
    defensive = attribution.loc[attribution["final_position"] < 0.995].copy()
    best_defensive = defensive.loc[defensive["buy_hold_return"] < 0].sort_values(
        "position_timing_contribution",
        ascending=False,
    ).head(10)
    worst_defensive = defensive.loc[defensive["buy_hold_return"] > 0].sort_values(
        "position_timing_contribution",
    ).head(10)

    def table(df: pd.DataFrame) -> str:
        cols = [
            "date",
            "final_position",
            "buy_hold_return",
            "position_timing_contribution",
            "ensemble_upside_score",
        ]
        use_cols = [col for col in cols if col in df.columns]
        if df.empty or not use_cols:
            return "No rows."
        use = df.loc[:, use_cols]
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

    component_lines = []
    for idx, component in enumerate(selected_components, start=1):
        penalties = component.get("penalty_breakdown", {})
        component_lines.append(
            "- Component "
            f"{idx}: {component['model_name']} {component['horizon']}D {component['signal_name']}; "
            f"stable_score={_fmt(component.get('stable_score'))}; "
            f"raw_score={_fmt(component.get('raw_stable_score'))}; "
            f"penalty={_fmt(component.get('complexity_penalty'))}; "
            f"penalty_breakdown={penalties}"
        )
    success = summary["avoided_downside"] > summary["missed_upside"] + summary["transaction_cost"]
    conclusion = (
        "The strategy avoided downside more than missed upside and transaction cost."
        if success
        else "The strategy did not beat buy-and-hold because missed upside remains larger than avoided downside; the main limitation is still signal generalization, not transaction cost."
    )
    text = f"""# Stability-Aware High-Participation Attribution Summary

## Selected Signal Components
{chr(10).join(component_lines)}

## Summary
- Total days below full exposure: {int(summary["days_below_full_exposure"])}
- Percentage of days below full exposure: {_fmt(summary["pct_days_below_full_exposure"])}
- Average position: {_fmt(summary["avg_position"])}
- Minimum position: {_fmt(summary["min_position"])}
- Maximum position: {_fmt(summary["max_position"])}
- Total turnover: {_fmt(summary["total_turnover"])}
- Missed upside from reduced exposure on positive-return days: {_fmt(summary["missed_upside"])}
- Avoided downside from reduced exposure on negative-return days: {_fmt(summary["avoided_downside"])}
- Transaction cost: {_fmt(summary["transaction_cost"])}
- Net position timing contribution: {_fmt(summary["net_position_timing_contribution"])}

## Conclusion
{conclusion}

## Top 10 Best Defensive Days
{table(best_defensive)}

## Top 10 Worst Defensive Days
{table(worst_defensive)}
"""
    path.write_text(text, encoding="utf-8")


def build_risk_budget_attribution(
    strategy_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    df = strategy_daily.loc[strategy_daily["strategy"] == RISK_BUDGET_STRATEGY_NAME].sort_values("date").copy()
    if df.empty:
        raise ValueError(f"No strategy rows for {RISK_BUDGET_STRATEGY_NAME}")
    position_gap = (1.0 - df["final_position"].astype(float)).clip(lower=0.0)
    ret = df["buy_hold_return"].astype(float)
    df["position_gap_from_full"] = position_gap
    df["position_timing_contribution"] = position_gap * (-ret)
    missed_upside = float((position_gap * ret.clip(lower=0.0)).sum())
    avoided_downside = float((position_gap * (-ret.clip(upper=0.0))).sum())
    transaction_cost = float(df["transaction_cost"].sum())
    max_drawdown = float(df["drawdown"].min())
    benchmark_max_drawdown = float(df["buy_hold_drawdown"].min())
    summary = {
        "days_below_full_exposure": int((df["final_position"] < 0.995).sum()),
        "avg_position": float(df["final_position"].mean()),
        "min_position": float(df["final_position"].min()),
        "max_position": float(df["final_position"].max()),
        "total_turnover": float(df["turnover"].sum()),
        "max_drawdown": max_drawdown,
        "benchmark_max_drawdown": benchmark_max_drawdown,
        "drawdown_improvement_vs_buy_hold": max_drawdown - benchmark_max_drawdown,
        "missed_upside": missed_upside,
        "avoided_downside": avoided_downside,
        "transaction_cost": transaction_cost,
        "net_position_timing_contribution": avoided_downside - missed_upside - transaction_cost,
    }
    return df, summary


def write_risk_budget_attribution_summary(
    strategy_daily: pd.DataFrame,
    path: Path,
) -> None:
    attribution, summary = build_risk_budget_attribution(strategy_daily)
    defensive = attribution.loc[attribution["final_position"] < 0.995].copy()
    best_defensive = defensive.loc[defensive["buy_hold_return"] < 0].sort_values(
        "position_timing_contribution",
        ascending=False,
    ).head(10)
    worst_defensive = defensive.loc[defensive["buy_hold_return"] > 0].sort_values(
        "position_timing_contribution",
    ).head(10)

    def table(df: pd.DataFrame) -> str:
        cols = [
            "date",
            "final_position",
            "buy_hold_return",
            "position_timing_contribution",
            "risk_score",
            "pred_up_prob",
            "pred_ret",
        ]
        use_cols = [col for col in cols if col in df.columns]
        if df.empty or not use_cols:
            return "No rows."
        use = df.loc[:, use_cols]
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

    reduces_drawdown = summary["drawdown_improvement_vs_buy_hold"] > 0
    timing_success = summary["avoided_downside"] > summary["missed_upside"] + summary["transaction_cost"]
    conclusion = (
        "The risk-budget framework reduced realized maximum drawdown in the test period."
        if reduces_drawdown
        else "The risk-budget framework did not reduce realized maximum drawdown relative to buy-and-hold."
    )
    contribution = (
        "Avoided downside is larger than missed upside and transaction cost."
        if timing_success
        else "Missed upside remains larger than avoided downside and transaction cost."
    )
    text = f"""# Risk-Budget Attribution Summary

## Summary
- Total days below full exposure: {int(summary["days_below_full_exposure"])}
- Average position: {_fmt(summary["avg_position"])}
- Minimum position: {_fmt(summary["min_position"])}
- Maximum position: {_fmt(summary["max_position"])}
- Total turnover: {_fmt(summary["total_turnover"])}
- Max drawdown: {_fmt(summary["max_drawdown"])}
- Buy-and-hold max drawdown: {_fmt(summary["benchmark_max_drawdown"])}
- Drawdown improvement vs buy-and-hold: {_fmt(summary["drawdown_improvement_vs_buy_hold"])}
- Missed upside from reduced exposure on positive-return days: {_fmt(summary["missed_upside"])}
- Avoided downside from reduced exposure on negative-return days: {_fmt(summary["avoided_downside"])}
- Transaction cost: {_fmt(summary["transaction_cost"])}
- Net position timing contribution: {_fmt(summary["net_position_timing_contribution"])}

## Conclusion
{conclusion} {contribution}

## Top 10 Best Defensive Days
{table(best_defensive)}

## Top 10 Worst Defensive Days
{table(worst_defensive)}
"""
    path.write_text(text, encoding="utf-8")


def build_early_stress_risk_budget_attribution(
    strategy_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float | int | str | bool]]:
    df = strategy_daily.loc[strategy_daily["strategy"] == EARLY_STRESS_STRATEGY_NAME].sort_values("date").copy()
    if df.empty:
        raise ValueError(f"No strategy rows for {EARLY_STRESS_STRATEGY_NAME}")
    benchmark = strategy_daily.loc[strategy_daily["strategy"] == "buy_hold"].sort_values("date").copy()
    trough_date = pd.NaT
    if not benchmark.empty:
        trough_date = pd.to_datetime(benchmark.loc[benchmark["buy_hold_drawdown"].idxmin(), "date"])
    position_gap = (1.0 - df["final_position"].astype(float)).clip(lower=0.0)
    ret = df["buy_hold_return"].astype(float)
    df["position_gap_from_full"] = position_gap
    df["position_timing_contribution"] = position_gap * (-ret)
    missed_upside = float((position_gap * ret.clip(lower=0.0)).sum())
    avoided_downside = float((position_gap * (-ret.clip(upper=0.0))).sum())
    transaction_cost = float(df["transaction_cost"].sum())
    early_signal = df.get("early_stress_signal", pd.Series(0.0, index=df.index)).fillna(0.0).astype(float) > 0
    pre_trough_signal = df.loc[early_signal & (pd.to_datetime(df["date"]) < trough_date)] if pd.notna(trough_date) else pd.DataFrame()
    first_pre_trough_signal = pre_trough_signal.iloc[0]["date"] if not pre_trough_signal.empty else pd.NaT
    summary: dict[str, float | int | str | bool] = {
        "days_below_full_exposure": int((df["final_position"] < 0.995).sum()),
        "avg_position": float(df["final_position"].mean()),
        "min_position": float(df["final_position"].min()),
        "max_position": float(df["final_position"].max()),
        "total_turnover": float(df["turnover"].sum()),
        "missed_upside": missed_upside,
        "avoided_downside": avoided_downside,
        "transaction_cost": transaction_cost,
        "net_position_timing_contribution": avoided_downside - missed_upside - transaction_cost,
        "benchmark_trough_date": str(pd.Timestamp(trough_date).date()) if pd.notna(trough_date) else "NA",
        "early_stress_triggered_before_benchmark_trough": bool(not pre_trough_signal.empty),
        "first_early_stress_signal_before_trough": (
            str(pd.Timestamp(first_pre_trough_signal).date()) if pd.notna(first_pre_trough_signal) else "NA"
        ),
    }
    return df, summary


def write_early_stress_risk_budget_attribution_summary(
    strategy_daily: pd.DataFrame,
    path: Path,
) -> None:
    attribution, summary = build_early_stress_risk_budget_attribution(strategy_daily)
    defensive = attribution.loc[attribution["final_position"] < 0.995].copy()
    best_defensive = defensive.loc[defensive["buy_hold_return"] < 0].sort_values(
        "position_timing_contribution",
        ascending=False,
    ).head(10)
    worst_defensive = defensive.loc[defensive["buy_hold_return"] > 0].sort_values(
        "position_timing_contribution",
    ).head(10)

    def table(df: pd.DataFrame) -> str:
        cols = [
            "date",
            "final_position",
            "buy_hold_return",
            "position_timing_contribution",
            "early_stress_score",
            "pred_up_prob",
            "pred_big_up_prob",
        ]
        use_cols = [col for col in cols if col in df.columns]
        if df.empty or not use_cols:
            return "No rows."
        use = df.loc[:, use_cols]
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

    timing_success = float(summary["avoided_downside"]) > float(summary["missed_upside"]) + float(summary["transaction_cost"])
    conclusion = (
        "Avoided downside is larger than missed upside and transaction cost."
        if timing_success
        else "Missed upside remains larger than avoided downside and transaction cost."
    )
    text = f"""# Early-Stress Risk-Budget Attribution Summary

## Summary
- Total days below full exposure: {int(summary["days_below_full_exposure"])}
- Average position: {_fmt(summary["avg_position"])}
- Minimum position: {_fmt(summary["min_position"])}
- Maximum position: {_fmt(summary["max_position"])}
- Total turnover: {_fmt(summary["total_turnover"])}
- Missed upside from reduced exposure on positive-return days: {_fmt(summary["missed_upside"])}
- Avoided downside from reduced exposure on negative-return days: {_fmt(summary["avoided_downside"])}
- Transaction cost: {_fmt(summary["transaction_cost"])}
- Net position timing contribution: {_fmt(summary["net_position_timing_contribution"])}
- Benchmark trough date: {summary["benchmark_trough_date"]}
- Early stress signals triggered before benchmark trough: {"Yes" if summary["early_stress_triggered_before_benchmark_trough"] else "No"}
- First early stress signal before trough: {summary["first_early_stress_signal_before_trough"]}

## Conclusion
{conclusion}

## Top 10 Best Defensive Days
{table(best_defensive)}

## Top 10 Worst Defensive Days
{table(worst_defensive)}
"""
    path.write_text(text, encoding="utf-8")
