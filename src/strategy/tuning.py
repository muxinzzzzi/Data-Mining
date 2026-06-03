from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.backtest import backtest_position_frame, build_buy_hold_frame
from src.strategy.decision_rules import StrategyFamily, StrategyParams, build_strategy_positions, prepare_strategy_frame
from src.strategy.metrics import (
    compute_strategy_metrics,
    conservative_validation_strategy_score,
    ultra_conservative_validation_strategy_score,
    validation_strategy_score,
)


def candidate_signal_sources(validation_metrics: pd.DataFrame, limit: int = 5) -> list[tuple[str, int]]:
    metrics = validation_metrics.copy()
    if metrics.empty:
        raise ValueError("Validation metrics are empty.")
    metrics["strategy_source_score"] = (
        0.44 * metrics["big_up_AUC"].fillna(0.5)
        + 0.18 * metrics["trade_AUC"].fillna(0.5)
        + 0.16 * metrics["clean_direction_AUC"].fillna(0.5)
        + 0.12 * metrics["AUC"].fillna(0.5)
        + 0.10 * metrics["return_correlation"].clip(lower=0).fillna(0.0)
    )
    out = (
        metrics.sort_values("strategy_source_score", ascending=False)
        .drop_duplicates(["model_name", "horizon"])
        .head(limit)
    )
    return [(str(row.model_name), int(row.horizon)) for row in out.itertuples()]


def iter_param_candidates(sources: list[tuple[str, int]]) -> list[StrategyParams]:
    params: list[StrategyParams] = []
    grid = product(
        [0.56, 0.62],
        [0.58, 0.64],
        [0.00, 0.03],
        [0.995, 0.99],
        [0.99, 0.985],
        [0.98, 0.975],
        [0.01, 0.02],
        [1.10, 1.25],
        [0.06, 0.10],
    )
    grid_values = list(grid)
    for model_name, horizon in sources:
        for values in grid_values:
            params.append(StrategyParams(model_name, horizon, *values))
    return params


def iter_conservative_param_candidates(sources: list[tuple[str, int]]) -> list[StrategyParams]:
    params: list[StrategyParams] = []
    grid = product(
        [0.58, 0.62, 0.66],
        [0.02, 0.04, 0.06],
        [0.998, 0.995],
        [0.99, 0.985],
        [0.98, 0.975],
        [0.03, 0.05, 0.08],
    )
    for model_name, horizon in sources:
        for big_down, tail_score, mild_cut, risk_cut, severe_cut, no_trade_band in grid:
            params.append(
                StrategyParams(
                    model_name=model_name,
                    horizon=horizon,
                    big_up_prob_threshold=0.56,
                    big_down_prob_threshold=big_down,
                    tail_score_threshold=tail_score,
                    mild_cut_exposure=mild_cut,
                    defensive_cut_exposure=risk_cut,
                    severe_defensive_exposure=severe_cut,
                    no_trade_band=no_trade_band,
                    high_volatility_multiplier=1.20,
                    drawdown_threshold=0.10,
                )
            )
    return params


def iter_ultra_conservative_param_candidates(sources: list[tuple[str, int]]) -> list[StrategyParams]:
    params: list[StrategyParams] = []
    grid = product(
        [0.62, 0.66, 0.70],
        [0.04, 0.06, 0.08],
        [0.999, 0.998, 0.995],
        [0.995, 0.99, 0.985],
        [0.985, 0.98],
        [0.03, 0.05, 0.08],
    )
    for model_name, horizon in sources:
        for big_down, tail_score, mild_cut, risk_cut, severe_cut, no_trade_band in grid:
            params.append(
                StrategyParams(
                    model_name=model_name,
                    horizon=horizon,
                    big_up_prob_threshold=0.56,
                    big_down_prob_threshold=big_down,
                    tail_score_threshold=tail_score,
                    mild_cut_exposure=mild_cut,
                    defensive_cut_exposure=risk_cut,
                    severe_defensive_exposure=severe_cut,
                    no_trade_band=no_trade_band,
                    high_volatility_multiplier=1.20,
                    drawdown_threshold=0.10,
                )
            )
    return params


def tune_strategy_family(
    feature_df: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    family: StrategyFamily,
) -> dict[str, Any]:
    sources = candidate_signal_sources(validation_metrics)
    candidates = (
        iter_ultra_conservative_param_candidates(sources)
        if family == "ml_ultra_conservative_full_participation_enhancement"
        else (
            iter_conservative_param_candidates(sources)
            if family == "ml_conservative_full_participation_enhancement"
            else iter_param_candidates(sources)
        )
    )
    source_frames: dict[tuple[str, int], pd.DataFrame] = {}
    source_benchmarks: dict[tuple[str, int], pd.DataFrame] = {}
    for source in sources:
        dummy = StrategyParams(source[0], source[1], 0.56, 0.58, 0.0, 0.95, 0.90, 0.85, 0.01, 1.10, 0.06)
        frame = prepare_strategy_frame(feature_df, validation_predictions, dummy)
        source_frames[source] = frame
        source_benchmarks[source] = build_buy_hold_frame(frame)
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    for params in candidates:
        try:
            source = (params.model_name, params.horizon)
            frame = source_frames[source]
            position, signal = build_strategy_positions(frame, params, family)
            strategy_daily = backtest_position_frame(frame, position, family, signal=signal, initial_position=1.0)
            benchmark = source_benchmarks[source]
            combined = pd.concat([benchmark, strategy_daily], ignore_index=True)
            metrics = compute_strategy_metrics(combined)
            row = metrics.loc[metrics["strategy"] == family].iloc[0].to_dict()
        except Exception:
            continue

        score = (
            ultra_conservative_validation_strategy_score(pd.Series(row))
            if family == "ml_ultra_conservative_full_participation_enhancement"
            else (
                conservative_validation_strategy_score(pd.Series(row))
                if family == "ml_conservative_full_participation_enhancement"
                else validation_strategy_score(pd.Series(row))
            )
        )
        row_summary = {
            "family": family,
            "validation_score": score,
            "params": params.to_dict(),
            "total_return": row.get("total_return"),
            "excess_return_vs_buy_hold": row.get("excess_return_vs_buy_hold"),
            "max_drawdown": row.get("max_drawdown"),
            "sharpe": row.get("sharpe"),
            "total_turnover": row.get("total_turnover"),
            "benchmark_clone": row.get("benchmark_clone"),
            "days_below_full_exposure": row.get("days_below_full_exposure"),
            "average_position": row.get("average_position"),
            "missed_upside": row.get("missed_upside"),
            "avoided_downside": row.get("avoided_downside"),
            "net_timing_contribution": row.get("net_timing_contribution"),
        }
        rows.append(row_summary)
        if best is None or score > best["validation_score"]:
            best = row_summary

    if best is None:
        raise RuntimeError(f"No valid validation strategy candidates for {family}.")

    ranked_rows = sorted(rows, key=lambda item: item["validation_score"], reverse=True)
    return {
        "family": family,
        "selected": best,
        "selected_params": StrategyParams(**best["params"]),
        "candidate_count": len(rows),
        "top_candidates": ranked_rows[:20],
        "source_selection_rule": "Signal sources and parameters selected on validation period only.",
    }


def tune_all_strategy_families(
    feature_df: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    validation_metrics: pd.DataFrame,
) -> dict[str, Any]:
    families: list[StrategyFamily] = [
        "ml_big_up_index_enhancement",
        "ml_conservative_full_participation_enhancement",
        "ml_ultra_conservative_full_participation_enhancement",
        "ml_direct_signal_timing",
        "ml_big_up_plus_115",
        "ml_big_up_plus_120",
    ]
    return {
        family: tune_strategy_family(feature_df, validation_predictions, validation_metrics, family)
        for family in families
    }
