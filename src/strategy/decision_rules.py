from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.external_market_features import EXTERNAL_STRATEGY_CONTEXT_COLUMNS


StrategyFamily = Literal[
    "ml_big_up_index_enhancement",
    "ml_conservative_full_participation_enhancement",
    "ml_ultra_conservative_full_participation_enhancement",
    "ml_direct_signal_timing",
    "ml_upside_participation_enhancement",
    "ml_stability_aware_high_participation",
    "ml_risk_budget_enhancement",
    "ml_early_stress_risk_budget",
    "ml_external_risk_budget_enhancement_index_only",
    "ml_external_risk_budget_enhancement_index_margin",
    "ml_external_risk_budget_enhancement_index_shibor",
    "ml_external_risk_budget_enhancement_all",
    "ml_external_early_stress_risk_budget_index_only",
    "ml_external_early_stress_risk_budget_all",
    "ml_external_soft_confirm_risk_budget",
    "ml_external_soft_confirm_risk_budget_index_only",
    "ml_external_soft_confirm_early_stress",
    "ml_external_soft_confirm_early_stress_index_only",
    "ml_external_soft_lite_risk_budget",
    "ml_external_soft_lite_risk_budget_index_only",
    "ml_external_soft_lite_early_stress",
    "ml_external_soft_lite_early_stress_index_only",
    "ml_big_up_plus_115",
    "ml_big_up_plus_120",
]


@dataclass(frozen=True)
class StrategyParams:
    model_name: str
    horizon: int
    big_up_prob_threshold: float
    big_down_prob_threshold: float
    tail_score_threshold: float
    mild_cut_exposure: float
    defensive_cut_exposure: float
    severe_defensive_exposure: float
    no_trade_band: float
    high_volatility_multiplier: float
    drawdown_threshold: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class UpsideParticipationParams:
    model_name: str
    horizon: int
    signal_name: str
    high_signal_cut: float
    low_signal_cut: float
    neutral_position: float
    mild_cut_position: float
    risk_cut_position: float
    min_position: float
    strong_trend_floor: float
    high_vol_multiplier: float
    drawdown_cut: float
    no_trade_band: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class StabilityAwareHighParticipationParams:
    model_name: str
    horizon: int
    low_score_cut: float
    very_low_score_cut: float
    mild_cut_position: float
    risk_cut_position: float
    min_position: float
    high_vol_multiplier: float
    drawdown_cut: float
    no_trade_band: float
    require_risk_confirm: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class RiskBudgetParams:
    model_name: str
    horizon: int
    high_vol_multiplier: float
    drawdown_cut: float
    mild_risk_cut: float
    high_risk_cut: float
    severe_risk_cut: float
    mild_position: float
    high_risk_position: float
    severe_position: float
    min_position: float
    positive_prob: float
    big_up_prob_cut: float
    recovery_position: float
    no_trade_band: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class EarlyStressRiskBudgetParams:
    model_name: str
    horizon: int
    early_stress_cut: float
    severe_stress_cut: float
    weak_prob: float
    strong_up_prob: float
    big_up_prob_cut: float
    mild_position: float
    stress_position: float
    severe_position: float
    min_position: float
    no_trade_band: float
    require_ml_not_positive: bool
    recovery_fast: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalRiskBudgetParams(RiskBudgetParams):
    external_feature_set: str


@dataclass(frozen=True)
class ExternalEarlyStressRiskBudgetParams(EarlyStressRiskBudgetParams):
    external_feature_set: str


@dataclass(frozen=True)
class ExternalSoftConfirmRiskBudgetParams(RiskBudgetParams):
    external_feature_set: str
    high_threshold: float
    extreme_threshold: float
    high_unconfirmed_weight: float
    high_confirm_weight: float
    extreme_unconfirmed_weight: float
    extreme_confirm_weight: float
    extreme_stress_weight: float
    soft_min_weight: float
    style_floor: float
    strong_style_floor: float
    require_persistence: bool
    persistence_days: int


@dataclass(frozen=True)
class ExternalSoftConfirmEarlyStressParams(EarlyStressRiskBudgetParams):
    external_feature_set: str
    high_threshold: float
    extreme_threshold: float
    high_unconfirmed_weight: float
    high_confirm_weight: float
    extreme_unconfirmed_weight: float
    extreme_confirm_weight: float
    extreme_stress_weight: float
    soft_min_weight: float
    style_floor: float
    strong_style_floor: float
    require_persistence: bool
    persistence_days: int


FEATURE_CONTEXT_COLS = [
    "date",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "downside_return_sum_5",
    "downside_return_sum_10",
    "downside_return_sum_20",
    "large_down_count_5",
    "large_down_count_10",
    "negative_return_streak",
    "ret_5_vs_volatility_20",
    "downside_acceleration_5_20",
    "ma_ratio_20",
    "ma_ratio_60",
    "ma_spread_20_60",
    "volatility_20",
    "volatility_60",
    "volatility_ratio_5_20",
    "volatility_ratio_20_60",
    "drawdown_20",
    "drawdown_60",
    "drawdown_120",
    "drawdown_speed_20",
    "trend_regime_score",
    "high_volatility_regime",
    "drawdown_regime",
    "severe_drawdown_regime",
    "bear_high_vol_regime",
    "crash_risk_score",
    "crash_risk_score_roll60_70pct",
    "m5_last_hour_return_mean_3",
    "m5_last30_return_mean_3",
    "m5_negative_bar_ratio_mean_3",
    "m5_large_drop_count_mean_3",
    "m5_close_below_vwap_count_5",
    "m5_close_vwap_position_mean_3",
    "m5_intraday_sell_pressure_score",
    "sell_pressure_volume_5",
    "sell_pressure_volume_10",
    "down_day_volume_ratio_5",
    "down_day_volume_ratio_10",
    "volume_price_confirmation_5",
    "volume_price_confirmation_10",
    "abnormal_down_volume_5",
    "liquidity_stress_score_5",
    "early_downside_acceleration_score",
    "early_intraday_sell_pressure_score",
    "early_liquidity_stress_score",
    "early_weak_trend_score",
    "early_volatility_expansion_score",
    "early_stress_score",
    "m5_available",
    *EXTERNAL_STRATEGY_CONTEXT_COLUMNS,
]


def prepare_strategy_frame(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    params: StrategyParams
    | UpsideParticipationParams
    | StabilityAwareHighParticipationParams
    | RiskBudgetParams
    | EarlyStressRiskBudgetParams
    | ExternalRiskBudgetParams
    | ExternalEarlyStressRiskBudgetParams
    | ExternalSoftConfirmRiskBudgetParams
    | ExternalSoftConfirmEarlyStressParams,
) -> pd.DataFrame:
    pred = predictions.loc[
        (predictions["model_name"] == params.model_name) & (predictions["horizon"] == params.horizon)
    ].copy()
    if pred.empty:
        raise ValueError(f"No predictions for {params.model_name}, horizon={params.horizon}")
    context_base = feature_df.sort_values("date").copy()
    if "crash_risk_score" in context_base.columns:
        context_base["crash_risk_score_roll60_70pct"] = (
            context_base["crash_risk_score"].shift(1).rolling(60, min_periods=20).quantile(0.70)
        )
    context_cols = [col for col in FEATURE_CONTEXT_COLS if col in context_base.columns]
    context = context_base[context_cols].copy()
    frame = pred.merge(context, on="date", how="left")
    frame = frame.sort_values("date").dropna(subset=["fwd_ret_1d"]).reset_index(drop=True)
    return frame


def apply_no_trade_band(raw_position: pd.Series, no_trade_band: float, initial_position: float = 1.0) -> pd.Series:
    values: list[float] = []
    current = initial_position
    for value in raw_position.fillna(initial_position).astype(float):
        value = float(value)
        if not values or abs(value - current) >= no_trade_band:
            current = value
        values.append(current)
    return pd.Series(values, index=raw_position.index, dtype=float)


def apply_no_trade_band_with_full_reset(
    raw_position: pd.Series,
    no_trade_band: float,
    initial_position: float = 1.0,
) -> pd.Series:
    values: list[float] = []
    current = initial_position
    for value in raw_position.fillna(initial_position).astype(float):
        value = float(value)
        if value >= 0.999 and current < 0.999:
            current = 1.0
        elif not values or abs(value - current) >= no_trade_band:
            current = value
        values.append(current)
    return pd.Series(values, index=raw_position.index, dtype=float)


def _risk_flags(frame: pd.DataFrame, params: StrategyParams) -> tuple[pd.Series, pd.Series, pd.Series]:
    trend_score = frame.get("trend_regime_score", pd.Series(0.5, index=frame.index)).fillna(0.5)
    ret_20 = frame.get("ret_20", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ma_ratio_20 = frame.get("ma_ratio_20", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ma_ratio_60 = frame.get("ma_ratio_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    vol_ratio = frame.get("volatility_ratio_20_60", pd.Series(1.0, index=frame.index)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    drawdown_60 = frame.get("drawdown_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    drawdown_120 = frame.get("drawdown_120", pd.Series(0.0, index=frame.index)).fillna(0.0)
    high_vol_regime = frame.get("high_volatility_regime", pd.Series(0.0, index=frame.index)).fillna(0.0)
    severe_drawdown_regime = frame.get("severe_drawdown_regime", pd.Series(0.0, index=frame.index)).fillna(0.0)
    crash_risk = frame.get("crash_risk_score", pd.Series(0.0, index=frame.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    weak_trend = (trend_score < 0.35) | (ret_20 < 0) | ((ma_ratio_20 < 0) & (ma_ratio_60 < 0))
    high_vol = (vol_ratio > params.high_volatility_multiplier) | (high_vol_regime > 0)
    drawdown_risk = drawdown_60 < -abs(params.drawdown_threshold)
    severe_risk = (
        (drawdown_120 < -1.5 * abs(params.drawdown_threshold))
        | (severe_drawdown_regime > 0)
        | ((crash_risk > 0.12) & high_vol)
    )
    return weak_trend.fillna(False), (high_vol | drawdown_risk).fillna(False), severe_risk.fillna(False)


def big_up_index_enhancement_position(frame: pd.DataFrame, params: StrategyParams) -> pd.Series:
    big_up = frame["pred_big_up_prob"].fillna(0.5)
    big_down = frame["pred_big_down_prob"].fillna(0.5)
    tail = frame["pred_tail_score"].fillna(0.0)
    weak_trend, risk, severe_risk = _risk_flags(frame, params)

    positive = (big_up >= params.big_up_prob_threshold) | (tail >= params.tail_score_threshold)
    downside = (big_down >= params.big_down_prob_threshold) | (tail <= -params.tail_score_threshold)
    strong_positive = positive & ~weak_trend & ~risk & ~severe_risk
    mild_risk = (weak_trend | risk | downside) & ~severe_risk
    defensive = (downside & (weak_trend | risk)) & ~severe_risk

    raw = pd.Series(0.98, index=frame.index, dtype=float)
    raw = raw.mask(strong_positive, 1.00)
    raw = raw.mask(mild_risk, params.mild_cut_exposure)
    raw = raw.mask(defensive, params.defensive_cut_exposure)
    raw = raw.mask(severe_risk, params.severe_defensive_exposure)
    raw = raw.clip(lower=0.0, upper=1.0)
    return apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0)


def direct_signal_timing_position(frame: pd.DataFrame, params: StrategyParams) -> tuple[pd.Series, pd.Series]:
    big_up = frame["pred_big_up_prob"].fillna(0.5)
    big_down = frame["pred_big_down_prob"].fillna(0.5)
    clean_prob = frame["pred_clean_direction_prob"].fillna(0.5)
    up_prob = frame["pred_up_prob"].fillna(0.5)
    tail = frame["pred_tail_score"].fillna(0.0)
    weak_trend, risk, severe_risk = _risk_flags(frame, params)

    buy = ((big_up >= params.big_up_prob_threshold) | (tail >= params.tail_score_threshold)) & (
        (clean_prob >= 0.52) | (up_prob >= 0.52)
    )
    sell = ((big_down >= params.big_down_prob_threshold) | (tail <= -params.tail_score_threshold)) & (weak_trend | risk)
    sell = sell | severe_risk

    raw = pd.Series(0.80, index=frame.index, dtype=float)
    signal = pd.Series("HOLD", index=frame.index, dtype=object)
    raw = raw.mask(buy & ~sell, 1.00)
    signal = signal.mask(buy & ~sell, "BUY")
    raw = raw.mask(sell, params.defensive_cut_exposure)
    signal = signal.mask(sell, "SELL")
    raw = raw.mask(severe_risk, params.severe_defensive_exposure)
    signal = signal.mask(severe_risk, "SELL")
    raw = raw.clip(lower=0.0, upper=1.0)
    return apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0), signal


def plus_exposure_position(frame: pd.DataFrame, params: StrategyParams, cap: float) -> pd.Series:
    base = big_up_index_enhancement_position(frame, params)
    big_up = frame["pred_big_up_prob"].fillna(0.5)
    tail = frame["pred_tail_score"].fillna(0.0)
    weak_trend, risk, severe_risk = _risk_flags(frame, params)

    very_positive = (big_up >= params.big_up_prob_threshold + 0.04) | (tail >= params.tail_score_threshold + 0.03)
    plus_allowed = very_positive & ~weak_trend & ~risk & ~severe_risk
    raw = base.copy()
    raw = raw.mask(plus_allowed, cap)
    raw = raw.mask(risk | weak_trend | severe_risk, np.minimum(raw, 1.0))
    raw = raw.clip(lower=0.0, upper=cap)
    return apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0)


def conservative_full_participation_position(frame: pd.DataFrame, params: StrategyParams) -> pd.Series:
    big_down = frame["pred_big_down_prob"].fillna(0.5)
    tail = frame["pred_tail_score"].fillna(0.0)
    pred_ret = frame["pred_ret"].fillna(0.0)
    up_prob = frame["pred_up_prob"].fillna(0.5)
    trend_score = frame.get("trend_regime_score", pd.Series(0.5, index=frame.index)).fillna(0.5)
    ma_ratio_60 = frame.get("ma_ratio_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    drawdown_60 = frame.get("drawdown_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    vol_20 = frame.get("volatility_20", pd.Series(np.nan, index=frame.index)).replace([np.inf, -np.inf], np.nan)
    vol_60 = frame.get("volatility_60", pd.Series(np.nan, index=frame.index)).replace([np.inf, -np.inf], np.nan)
    vol_ratio = (vol_20 / vol_60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    ml_negative = (
        (big_down >= params.big_down_prob_threshold)
        | (tail <= -abs(params.tail_score_threshold))
        | (pred_ret < 0)
        | (up_prob < 0.45)
    )
    weak_trend = (trend_score < 0.5) | (ma_ratio_60 < 0)
    high_risk = (drawdown_60 < -0.10) | (vol_ratio > params.high_volatility_multiplier)

    evidence_count = (
        (big_down >= params.big_down_prob_threshold).astype(int)
        + (tail <= -abs(params.tail_score_threshold)).astype(int)
        + (pred_ret < 0).astype(int)
        + (up_prob < 0.45).astype(int)
        + (trend_score < 0.5).astype(int)
        + (ma_ratio_60 < 0).astype(int)
        + (drawdown_60 < -0.10).astype(int)
        + (vol_ratio > params.high_volatility_multiplier).astype(int)
    )
    mild_cut = evidence_count >= 2
    risk_cut = ml_negative & weak_trend & high_risk
    severe_cut = risk_cut & (
        (evidence_count >= 5)
        | (drawdown_60 < -0.15)
        | (vol_ratio > 1.45)
        | (
            frame.get("severe_drawdown_regime", pd.Series(0.0, index=frame.index)).fillna(0.0)
            > 0
        )
    )

    raw = pd.Series(1.0, index=frame.index, dtype=float)
    raw = raw.mask(mild_cut, params.mild_cut_exposure)
    raw = raw.mask(risk_cut, params.defensive_cut_exposure)
    raw = raw.mask(severe_cut, params.severe_defensive_exposure)
    raw = raw.clip(lower=0.92, upper=1.0)
    return apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0)


def ultra_conservative_full_participation_position(frame: pd.DataFrame, params: StrategyParams) -> pd.Series:
    big_down = frame["pred_big_down_prob"].fillna(0.5)
    tail = frame["pred_tail_score"].fillna(0.0)
    pred_ret = frame["pred_ret"].fillna(0.0)
    up_prob = frame["pred_up_prob"].fillna(0.5)
    trend_score = frame.get("trend_regime_score", pd.Series(0.5, index=frame.index)).fillna(0.5)
    ma_ratio_60 = frame.get("ma_ratio_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    drawdown_60 = frame.get("drawdown_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    vol_20 = frame.get("volatility_20", pd.Series(np.nan, index=frame.index)).replace([np.inf, -np.inf], np.nan)
    vol_60 = frame.get("volatility_60", pd.Series(np.nan, index=frame.index)).replace([np.inf, -np.inf], np.nan)
    vol_ratio = (vol_20 / vol_60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    severe_drawdown_regime = frame.get("severe_drawdown_regime", pd.Series(0.0, index=frame.index)).fillna(0.0)
    crash_risk = frame.get("crash_risk_score", pd.Series(0.0, index=frame.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ml_negative_count = (
        (big_down >= params.big_down_prob_threshold).astype(int)
        + (tail <= -abs(params.tail_score_threshold)).astype(int)
        + (pred_ret < 0).astype(int)
        + (up_prob < 0.45).astype(int)
    )
    trend_negative_count = (
        (trend_score < 0.5).astype(int)
        + (ma_ratio_60 < 0).astype(int)
    )
    risk_negative_count = (
        (drawdown_60 < -0.10).astype(int)
        + (vol_ratio > params.high_volatility_multiplier).astype(int)
    )
    weak_signal_count = ml_negative_count + trend_negative_count + risk_negative_count

    ml_negative = ml_negative_count >= 1
    strong_ml_negative = (
        (ml_negative_count >= 2)
        | (big_down >= params.big_down_prob_threshold + 0.06)
        | (tail <= -abs(params.tail_score_threshold) - 0.04)
    )
    weak_trend = trend_negative_count >= 1
    high_risk = risk_negative_count >= 1
    stress = (
        (drawdown_60 < -0.14)
        | (vol_ratio > 1.40)
        | (severe_drawdown_regime > 0)
        | ((crash_risk > 0.16) & high_risk)
    )

    mild_cut = weak_signal_count >= 4
    defensive_cut = ml_negative & weak_trend & high_risk
    severe_cut = strong_ml_negative & weak_trend & high_risk & stress

    raw = pd.Series(1.0, index=frame.index, dtype=float)
    raw = raw.mask(mild_cut, params.mild_cut_exposure)
    raw = raw.mask(defensive_cut, params.defensive_cut_exposure)
    raw = raw.mask(severe_cut, params.severe_defensive_exposure)
    raw = raw.clip(lower=0.94, upper=1.0)
    return apply_no_trade_band_with_full_reset(raw, params.no_trade_band, initial_position=1.0)


def ml_upside_participation_enhancement_position(
    merged: pd.DataFrame,
    params: UpsideParticipationParams,
) -> pd.Series:
    if params.signal_name not in merged.columns:
        raise ValueError(f"Selected signal is not available in strategy frame: {params.signal_name}")

    selected_signal = merged[params.signal_name].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    ma_ratio_60 = merged.get("ma_ratio_60", pd.Series(0.0, index=merged.index)).fillna(0.0)
    ma_spread_20_60 = merged.get("ma_spread_20_60", pd.Series(0.0, index=merged.index)).fillna(0.0)
    ret_20 = merged.get("ret_20", pd.Series(0.0, index=merged.index)).fillna(0.0)
    trend_score = merged.get("trend_regime_score", pd.Series(0.5, index=merged.index)).fillna(0.5)
    drawdown_60 = merged.get("drawdown_60", pd.Series(0.0, index=merged.index)).fillna(0.0)
    volatility_20 = merged.get("volatility_20", pd.Series(np.nan, index=merged.index)).replace([np.inf, -np.inf], np.nan)
    volatility_60 = merged.get("volatility_60", pd.Series(np.nan, index=merged.index)).replace([np.inf, -np.inf], np.nan)
    bear_high_vol = merged.get("bear_high_vol_regime", pd.Series(0.0, index=merged.index)).fillna(0.0)
    crash_risk = merged.get("crash_risk_score", pd.Series(0.0, index=merged.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    crash_cut = merged.get(
        "crash_risk_score_roll60_70pct",
        pd.Series(np.nan, index=merged.index),
    ).replace([np.inf, -np.inf], np.nan)

    strong_upside = selected_signal >= params.high_signal_cut
    weak_upside = selected_signal <= params.low_signal_cut
    strong_trend = (ma_ratio_60 > 0) & (ma_spread_20_60 > 0) & (ret_20 > 0)
    weak_trend = (ma_ratio_60 < 0) | (ret_20 < 0) | (trend_score < 0.5)
    high_risk = (
        (drawdown_60 < params.drawdown_cut)
        | (volatility_20 > params.high_vol_multiplier * volatility_60)
        | (bear_high_vol > 0)
        | (crash_risk > crash_cut.fillna(np.inf))
    )

    raw = pd.Series(1.0, index=merged.index, dtype=float)
    raw = raw.mask(~strong_upside & ~weak_upside, params.neutral_position)
    raw = raw.mask(weak_upside & weak_trend, params.mild_cut_position)
    raw = raw.mask(weak_upside & weak_trend & high_risk, params.risk_cut_position)
    raw = raw.mask(strong_upside, 1.0)
    raw = raw.mask(strong_trend, np.maximum(raw, params.strong_trend_floor))
    raw = raw.clip(lower=params.min_position, upper=1.0)
    return apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0)


def ml_stability_aware_high_participation_position(
    merged: pd.DataFrame,
    params: StabilityAwareHighParticipationParams,
) -> pd.Series:
    if "ensemble_upside_score" not in merged.columns:
        raise ValueError("ensemble_upside_score is required for stability-aware high-participation strategy.")

    ensemble = merged["ensemble_upside_score"].replace([np.inf, -np.inf], np.nan).fillna(0.5)
    ma_ratio_60 = merged.get("ma_ratio_60", pd.Series(0.0, index=merged.index)).fillna(0.0)
    ma_spread_20_60 = merged.get("ma_spread_20_60", pd.Series(0.0, index=merged.index)).fillna(0.0)
    ret_20 = merged.get("ret_20", pd.Series(0.0, index=merged.index)).fillna(0.0)
    trend_score = merged.get("trend_regime_score", pd.Series(0.5, index=merged.index)).fillna(0.5)
    drawdown_60 = merged.get("drawdown_60", pd.Series(0.0, index=merged.index)).fillna(0.0)
    volatility_20 = merged.get("volatility_20", pd.Series(np.nan, index=merged.index)).replace([np.inf, -np.inf], np.nan)
    volatility_60 = merged.get("volatility_60", pd.Series(np.nan, index=merged.index)).replace([np.inf, -np.inf], np.nan)
    bear_high_vol = merged.get("bear_high_vol_regime", pd.Series(0.0, index=merged.index)).fillna(0.0)
    crash_risk = merged.get("crash_risk_score", pd.Series(0.0, index=merged.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    crash_cut = merged.get(
        "crash_risk_score_roll60_70pct",
        pd.Series(np.nan, index=merged.index),
    ).replace([np.inf, -np.inf], np.nan)

    weak_trend = (ma_ratio_60 < 0) | (ret_20 < 0) | (trend_score < 0.5)
    high_risk = (
        (drawdown_60 < params.drawdown_cut)
        | (volatility_20 > params.high_vol_multiplier * volatility_60)
        | (bear_high_vol > 0)
        | (crash_risk > crash_cut.fillna(np.inf))
    )
    strong_trend = (ma_ratio_60 > 0) & (ma_spread_20_60 > 0) & (ret_20 > 0)
    low_score = ensemble < params.low_score_cut
    very_low_score = ensemble < params.very_low_score_cut

    raw = pd.Series(1.0, index=merged.index, dtype=float)
    raw = raw.mask(low_score & weak_trend, params.mild_cut_position)
    raw = raw.mask(low_score & high_risk, params.risk_cut_position)
    severe_condition = very_low_score & weak_trend & high_risk
    if not params.require_risk_confirm:
        severe_condition = very_low_score & weak_trend
    raw = raw.mask(severe_condition, params.min_position)
    raw = raw.mask(strong_trend, np.maximum(raw, 0.99))
    raw = raw.clip(lower=params.min_position, upper=1.0)
    return apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0)


def ml_risk_budget_enhancement_position(
    merged: pd.DataFrame,
    params: RiskBudgetParams,
) -> pd.Series:
    ma_ratio_60 = merged.get("ma_ratio_60", pd.Series(0.0, index=merged.index)).fillna(0.0)
    ret_20 = merged.get("ret_20", pd.Series(0.0, index=merged.index)).fillna(0.0)
    trend_score = merged.get("trend_regime_score", pd.Series(0.5, index=merged.index)).fillna(0.5)
    drawdown_60 = merged.get("drawdown_60", pd.Series(0.0, index=merged.index)).fillna(0.0)
    volatility_20 = merged.get("volatility_20", pd.Series(np.nan, index=merged.index)).replace([np.inf, -np.inf], np.nan)
    volatility_60 = merged.get("volatility_60", pd.Series(np.nan, index=merged.index)).replace([np.inf, -np.inf], np.nan)
    bear_high_vol = merged.get("bear_high_vol_regime", pd.Series(0.0, index=merged.index)).fillna(0.0)
    crash_risk_value = merged.get("crash_risk_score", pd.Series(0.0, index=merged.index)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    crash_cut = merged.get(
        "crash_risk_score_roll60_70pct",
        pd.Series(np.nan, index=merged.index),
    ).replace([np.inf, -np.inf], np.nan)

    volatility_risk = (volatility_20 > params.high_vol_multiplier * volatility_60).fillna(False)
    drawdown_risk = (drawdown_60 < params.drawdown_cut).fillna(False)
    trend_risk = ((ma_ratio_60 < 0) | (ret_20 < 0) | (trend_score < 0.5)).fillna(False)
    crash_risk = (crash_risk_value > crash_cut.fillna(np.inf)).fillna(False)
    bear_high_vol_risk = (bear_high_vol > 0).fillna(False)
    risk_score = (
        0.30 * volatility_risk.astype(float)
        + 0.25 * drawdown_risk.astype(float)
        + 0.20 * trend_risk.astype(float)
        + 0.15 * crash_risk.astype(float)
        + 0.10 * bear_high_vol_risk.astype(float)
    )
    merged["risk_score"] = risk_score
    merged["volatility_risk"] = volatility_risk.astype(float)
    merged["drawdown_risk"] = drawdown_risk.astype(float)
    merged["trend_risk"] = trend_risk.astype(float)
    merged["crash_risk_flag"] = crash_risk.astype(float)
    merged["bear_high_vol_risk"] = bear_high_vol_risk.astype(float)

    raw = pd.Series(1.0, index=merged.index, dtype=float)
    raw = raw.mask(risk_score >= params.severe_risk_cut, params.severe_position)
    raw = raw.mask((risk_score >= params.high_risk_cut) & (risk_score < params.severe_risk_cut), params.high_risk_position)
    raw = raw.mask((risk_score >= params.mild_risk_cut) & (risk_score < params.high_risk_cut), params.mild_position)

    pred_up = merged.get("pred_up_prob", pd.Series(0.5, index=merged.index)).fillna(0.5)
    pred_ret = merged.get("pred_ret", pd.Series(0.0, index=merged.index)).fillna(0.0)
    pred_big_up = merged.get("pred_big_up_prob", pd.Series(0.5, index=merged.index)).fillna(0.5)
    pred_big_down = merged.get("pred_big_down_prob", pd.Series(0.5, index=merged.index)).fillna(0.5)
    recovery = ((pred_up >= params.positive_prob) & (pred_ret >= 0)) | (pred_big_up >= params.big_up_prob_cut)
    raw = raw.mask(recovery, np.maximum(raw, params.recovery_position))

    ml_negative = (pred_up < 0.45) & (pred_ret < 0) & (pred_big_down >= 0.50)
    raw = raw.mask(ml_negative & (risk_score >= params.high_risk_cut), np.minimum(raw, params.high_risk_position))
    raw = raw.mask(ml_negative & (risk_score >= params.severe_risk_cut), np.minimum(raw, params.severe_position))
    raw = raw.clip(lower=params.min_position, upper=1.0)
    return apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0)


def ml_early_stress_risk_budget_position(
    merged: pd.DataFrame,
    params: EarlyStressRiskBudgetParams,
) -> pd.Series:
    early_score = merged.get("early_stress_score", pd.Series(0.0, index=merged.index)).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)
    pred_up = merged.get("pred_up_prob", pd.Series(0.5, index=merged.index)).fillna(0.5)
    pred_big_up = merged.get("pred_big_up_prob", pd.Series(0.5, index=merged.index)).fillna(0.5)
    pred_ret = merged.get("pred_ret", pd.Series(0.0, index=merged.index)).fillna(0.0)

    early_stress = early_score >= params.early_stress_cut
    severe_stress = early_score >= params.severe_stress_cut
    ml_not_positive = (pred_up < params.strong_up_prob) & (pred_big_up < params.big_up_prob_cut)
    ml_weak = (pred_up < params.weak_prob) & (pred_ret < 0)
    strong_ml_upside = (pred_up >= params.strong_up_prob) | (pred_big_up >= params.big_up_prob_cut)
    stress_gate = ml_not_positive | ml_weak
    if not params.require_ml_not_positive:
        stress_gate = pd.Series(True, index=merged.index)

    raw = pd.Series(1.0, index=merged.index, dtype=float)
    raw = raw.mask(early_stress & ~stress_gate, params.mild_position)
    raw = raw.mask(early_stress & stress_gate, params.stress_position)
    raw = raw.mask(severe_stress & stress_gate, params.severe_position)
    raw = raw.mask(strong_ml_upside, 1.0)
    if params.recovery_fast:
        raw = raw.mask(early_score < params.early_stress_cut, 1.0)
    raw = raw.clip(lower=params.min_position, upper=1.0)

    merged["early_stress_signal"] = early_stress.astype(float)
    merged["severe_early_stress_signal"] = severe_stress.astype(float)
    merged["ml_not_positive_for_early_stress"] = ml_not_positive.astype(float)
    merged["ml_strong_upside_recovery"] = strong_ml_upside.astype(float)
    return apply_no_trade_band_with_full_reset(raw, params.no_trade_band, initial_position=1.0)


def _numeric_frame_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _external_score_for_feature_set(merged: pd.DataFrame, feature_set: str) -> pd.Series:
    index_score = _numeric_frame_series(merged, "external_index_risk_score", 0.0)
    margin_score = _numeric_frame_series(merged, "external_margin_risk_score", 0.0)
    shibor_score = _numeric_frame_series(merged, "external_shibor_risk_score", 0.0)
    market_score = _numeric_frame_series(merged, "external_market_risk_score", 0.0)
    if feature_set == "index_only":
        return index_score
    if feature_set == "index_margin":
        return index_score + margin_score
    if feature_set == "index_shibor":
        return index_score + shibor_score
    if feature_set == "all":
        return market_score
    raise ValueError(f"Unknown external feature set: {feature_set}")


def _external_overlay_weight(
    merged: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.Series, pd.Series]:
    score = _external_score_for_feature_set(merged, feature_set)
    weight = pd.Series(1.0, index=merged.index, dtype=float)
    weight = weight.mask(score == 2, 0.95)
    weight = weight.mask(score == 3, 0.90)
    weight = weight.mask(score >= 4, 0.80)

    liquidity_support = _numeric_frame_series(merged, "external_liquidity_support", 1.0) > 0
    style_support = _numeric_frame_series(merged, "external_style_support", 0.0) > 0
    pred_up = _numeric_frame_series(merged, "pred_up_prob", 0.5)
    pred_big_up = _numeric_frame_series(merged, "pred_big_up_prob", 0.5)
    pred_ret = _numeric_frame_series(merged, "pred_ret", 0.0)
    strong_ml_upside = (pred_up >= 0.55) | (pred_big_up >= 0.55) | ((pred_up >= 0.50) & (pred_ret >= 0))

    weight = weight.mask((score >= 3) & ~liquidity_support, np.minimum(weight, 0.85))
    weight = weight.mask(style_support & strong_ml_upside, np.maximum(weight, 0.95))
    weight = weight.clip(lower=0.0, upper=1.0)

    merged["external_overlay_score"] = score.astype(float)
    merged["external_overlay_weight"] = weight.astype(float)
    merged["external_overlay_feature_set"] = feature_set
    merged["external_overlay_style_support"] = style_support.astype(float)
    merged["external_overlay_liquidity_support"] = liquidity_support.astype(float)
    return score, weight


def ml_external_risk_budget_enhancement_position(
    merged: pd.DataFrame,
    params: ExternalRiskBudgetParams,
) -> pd.Series:
    base = ml_risk_budget_enhancement_position(merged, params)
    _, external_weight = _external_overlay_weight(merged, params.external_feature_set)
    raw = (base * external_weight).clip(lower=0.0, upper=1.0)
    merged["external_base_position"] = base.astype(float)
    return apply_no_trade_band_with_full_reset(raw, params.no_trade_band, initial_position=1.0)


def ml_external_early_stress_risk_budget_position(
    merged: pd.DataFrame,
    params: ExternalEarlyStressRiskBudgetParams,
) -> pd.Series:
    base = ml_early_stress_risk_budget_position(merged, params)
    _, external_weight = _external_overlay_weight(merged, params.external_feature_set)
    raw = (base * external_weight).clip(lower=0.0, upper=1.0)
    merged["external_base_position"] = base.astype(float)
    return apply_no_trade_band_with_full_reset(raw, params.no_trade_band, initial_position=1.0)


def _soft_external_internal_confirm(merged: pd.DataFrame, base_position: pd.Series) -> tuple[pd.Series, pd.Series]:
    early_score = _numeric_frame_series(merged, "early_stress_score", 0.0)
    ma_ratio_60 = _numeric_frame_series(merged, "ma_ratio_60", 0.0)
    ret_20 = _numeric_frame_series(merged, "ret_20", 0.0)
    ret_5 = _numeric_frame_series(merged, "ret_5", 0.0)
    trend_score = _numeric_frame_series(merged, "trend_regime_score", 0.5)
    volatility_20 = _numeric_frame_series(merged, "volatility_20", np.nan)
    volatility_60 = _numeric_frame_series(merged, "volatility_60", np.nan)
    drawdown_60 = _numeric_frame_series(merged, "drawdown_60", 0.0)
    drawdown_regime = _numeric_frame_series(merged, "drawdown_regime", 0.0)
    severe_drawdown = _numeric_frame_series(merged, "severe_drawdown_regime", 0.0)
    high_vol_regime = _numeric_frame_series(merged, "high_volatility_regime", 0.0)
    downside_accel = _numeric_frame_series(merged, "early_downside_acceleration_score", 0.0)
    intraday_pressure = _numeric_frame_series(merged, "early_intraday_sell_pressure_score", 0.0)
    liquidity_stress = _numeric_frame_series(merged, "early_liquidity_stress_score", 0.0)
    pred_up = _numeric_frame_series(merged, "pred_up_prob", 0.5)
    pred_big_up = _numeric_frame_series(merged, "pred_big_up_prob", 0.5)
    pred_ret = _numeric_frame_series(merged, "pred_ret", 0.0)

    vol_ratio = (volatility_20 / volatility_60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    trend_weak = (ma_ratio_60 < 0) | (ret_20 < 0) | (trend_score < 0.5)
    volatility_or_drawdown_risk = (
        (vol_ratio > 1.15)
        | (high_vol_regime > 0)
        | (drawdown_60 < -0.08)
        | (drawdown_regime > 0)
        | (severe_drawdown > 0)
    )
    strong_ml_upside = (pred_up >= 0.55) | (pred_big_up >= 0.55) | ((pred_up >= 0.50) & (pred_ret >= 0))
    internal_pressure_rising = (
        (early_score >= 0.35)
        | (downside_accel >= 0.45)
        | (intraday_pressure >= 0.45)
        | (liquidity_stress >= 0.45)
        | (ret_5 < 0)
    )
    internal_confirm = (
        (early_score >= 0.55)
        | (trend_weak & volatility_or_drawdown_risk)
        | (base_position < 0.98)
        | (~strong_ml_upside & internal_pressure_rising)
    )
    return internal_confirm.fillna(False), strong_ml_upside.fillna(False)


def _soft_external_weight(
    merged: pd.DataFrame,
    base_position: pd.Series,
    params: ExternalSoftConfirmRiskBudgetParams | ExternalSoftConfirmEarlyStressParams,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    score = _external_score_for_feature_set(merged, params.external_feature_set)
    internal_confirm, strong_ml_upside = _soft_external_internal_confirm(merged, base_position)
    early_score = _numeric_frame_series(merged, "early_stress_score", 0.0)
    high = score >= params.high_threshold
    extreme = score >= params.extreme_threshold

    weight = pd.Series(1.0, index=merged.index, dtype=float)
    weight = weight.mask(high & ~extreme & ~internal_confirm, params.high_unconfirmed_weight)
    weight = weight.mask(high & ~extreme & internal_confirm, params.high_confirm_weight)
    weight = weight.mask(extreme & ~internal_confirm, params.extreme_unconfirmed_weight)
    weight = weight.mask(extreme & internal_confirm, params.extreme_confirm_weight)
    weight = weight.mask(extreme & internal_confirm & (early_score >= 0.70), params.extreme_stress_weight)

    if params.require_persistence:
        risk_state = (high | extreme).astype(float)
        persisted = risk_state.rolling(params.persistence_days, min_periods=params.persistence_days).sum() >= params.persistence_days
        weight = weight.mask((high | extreme) & ~persisted, np.maximum(weight, 0.98))

    weight = weight.clip(lower=params.soft_min_weight, upper=1.0)
    style_support = (
        (_numeric_frame_series(merged, "external_style_support", 0.0) > 0)
        | (
            (_numeric_frame_series(merged, "external_small_vs_large_ret_20", 0.0) > 0)
            & (_numeric_frame_series(merged, "external_zz1000_ma_gap_20", 0.0) > 0)
        )
    )

    merged["external_soft_score"] = score.astype(float)
    merged["external_soft_weight"] = weight.astype(float)
    merged["external_soft_internal_confirm"] = internal_confirm.astype(float)
    merged["external_soft_strong_ml_upside"] = strong_ml_upside.astype(float)
    merged["external_soft_style_support"] = style_support.astype(float)
    merged["external_soft_state_high"] = high.astype(float)
    merged["external_soft_state_extreme"] = extreme.astype(float)
    return weight, style_support.fillna(False), strong_ml_upside.fillna(False)


def _apply_soft_style_recovery(
    raw_position: pd.Series,
    style_support: pd.Series,
    strong_ml_upside: pd.Series,
    score: pd.Series,
    params: ExternalSoftConfirmRiskBudgetParams | ExternalSoftConfirmEarlyStressParams,
) -> pd.Series:
    out = raw_position.copy()
    extreme = score >= params.extreme_threshold
    out = out.mask(style_support & strong_ml_upside, np.maximum(out, params.strong_style_floor))
    out = out.mask(style_support & ~extreme, np.maximum(out, params.style_floor))
    return out.clip(lower=0.0, upper=1.0)


def ml_external_soft_confirm_risk_budget_position(
    merged: pd.DataFrame,
    params: ExternalSoftConfirmRiskBudgetParams,
) -> pd.Series:
    base = ml_risk_budget_enhancement_position(merged, params)
    external_weight, style_support, strong_ml_upside = _soft_external_weight(merged, base, params)
    score = _external_score_for_feature_set(merged, params.external_feature_set)
    raw = (base * external_weight).clip(lower=0.0, upper=1.0)
    raw = _apply_soft_style_recovery(raw, style_support, strong_ml_upside, score, params)
    merged["external_soft_base_position"] = base.astype(float)
    return apply_no_trade_band_with_full_reset(raw, params.no_trade_band, initial_position=1.0)


def ml_external_soft_confirm_early_stress_position(
    merged: pd.DataFrame,
    params: ExternalSoftConfirmEarlyStressParams,
) -> pd.Series:
    base = ml_early_stress_risk_budget_position(merged, params)
    external_weight, style_support, strong_ml_upside = _soft_external_weight(merged, base, params)
    score = _external_score_for_feature_set(merged, params.external_feature_set)
    raw = (base * external_weight).clip(lower=0.0, upper=1.0)
    raw = _apply_soft_style_recovery(raw, style_support, strong_ml_upside, score, params)
    merged["external_soft_base_position"] = base.astype(float)
    return apply_no_trade_band_with_full_reset(raw, params.no_trade_band, initial_position=1.0)


def ml_external_soft_lite_risk_budget_position(
    merged: pd.DataFrame,
    params: ExternalSoftConfirmRiskBudgetParams,
) -> pd.Series:
    base = ml_risk_budget_enhancement_position(merged, params)
    external_weight, style_support, strong_ml_upside = _soft_external_weight(merged, base, params)
    score = _external_score_for_feature_set(merged, params.external_feature_set)
    raw = (base * external_weight).clip(lower=0.0, upper=1.0)
    raw = _apply_soft_style_recovery(raw, style_support, strong_ml_upside, score, params)
    merged["external_soft_lite_base_position"] = base.astype(float)
    return apply_no_trade_band_with_full_reset(raw, min(params.no_trade_band, 0.02), initial_position=1.0)


def ml_external_soft_lite_early_stress_position(
    merged: pd.DataFrame,
    params: ExternalSoftConfirmEarlyStressParams,
) -> pd.Series:
    base = ml_early_stress_risk_budget_position(merged, params)
    external_weight, style_support, strong_ml_upside = _soft_external_weight(merged, base, params)
    score = _external_score_for_feature_set(merged, params.external_feature_set)
    raw = (base * external_weight).clip(lower=0.0, upper=1.0)
    raw = _apply_soft_style_recovery(raw, style_support, strong_ml_upside, score, params)
    merged["external_soft_lite_base_position"] = base.astype(float)
    return apply_no_trade_band_with_full_reset(raw, min(params.no_trade_band, 0.02), initial_position=1.0)


def build_strategy_positions(
    frame: pd.DataFrame,
    params: StrategyParams
    | UpsideParticipationParams
    | StabilityAwareHighParticipationParams
    | RiskBudgetParams
    | EarlyStressRiskBudgetParams
    | ExternalRiskBudgetParams
    | ExternalEarlyStressRiskBudgetParams
    | ExternalSoftConfirmRiskBudgetParams
    | ExternalSoftConfirmEarlyStressParams,
    family: StrategyFamily,
) -> tuple[pd.Series, pd.Series]:
    if family == "ml_big_up_index_enhancement":
        return big_up_index_enhancement_position(frame, params), pd.Series("ADJUST", index=frame.index)  # type: ignore[arg-type]
    if family == "ml_conservative_full_participation_enhancement":
        return conservative_full_participation_position(frame, params), pd.Series("CONSERVATIVE", index=frame.index)  # type: ignore[arg-type]
    if family == "ml_ultra_conservative_full_participation_enhancement":
        return ultra_conservative_full_participation_position(frame, params), pd.Series("ULTRA_CONSERVATIVE", index=frame.index)  # type: ignore[arg-type]
    if family == "ml_direct_signal_timing":
        return direct_signal_timing_position(frame, params)  # type: ignore[arg-type]
    if family == "ml_upside_participation_enhancement":
        return ml_upside_participation_enhancement_position(frame, params), pd.Series("UPSIDE_PARTICIPATION", index=frame.index)  # type: ignore[arg-type]
    if family == "ml_stability_aware_high_participation":
        return ml_stability_aware_high_participation_position(frame, params), pd.Series("STABILITY_AWARE", index=frame.index)  # type: ignore[arg-type]
    if family == "ml_risk_budget_enhancement":
        return ml_risk_budget_enhancement_position(frame, params), pd.Series("RISK_BUDGET", index=frame.index)  # type: ignore[arg-type]
    if family == "ml_early_stress_risk_budget":
        return ml_early_stress_risk_budget_position(frame, params), pd.Series("EARLY_STRESS", index=frame.index)  # type: ignore[arg-type]
    if family in {
        "ml_external_risk_budget_enhancement_index_only",
        "ml_external_risk_budget_enhancement_index_margin",
        "ml_external_risk_budget_enhancement_index_shibor",
        "ml_external_risk_budget_enhancement_all",
    }:
        return ml_external_risk_budget_enhancement_position(frame, params), pd.Series("EXTERNAL_RISK_BUDGET", index=frame.index)  # type: ignore[arg-type]
    if family in {
        "ml_external_early_stress_risk_budget_index_only",
        "ml_external_early_stress_risk_budget_all",
    }:
        return ml_external_early_stress_risk_budget_position(frame, params), pd.Series("EXTERNAL_EARLY_STRESS", index=frame.index)  # type: ignore[arg-type]
    if family in {
        "ml_external_soft_confirm_risk_budget",
        "ml_external_soft_confirm_risk_budget_index_only",
    }:
        return ml_external_soft_confirm_risk_budget_position(frame, params), pd.Series("EXTERNAL_SOFT_RISK_BUDGET", index=frame.index)  # type: ignore[arg-type]
    if family in {
        "ml_external_soft_confirm_early_stress",
        "ml_external_soft_confirm_early_stress_index_only",
    }:
        return ml_external_soft_confirm_early_stress_position(frame, params), pd.Series("EXTERNAL_SOFT_EARLY_STRESS", index=frame.index)  # type: ignore[arg-type]
    if family in {
        "ml_external_soft_lite_risk_budget",
        "ml_external_soft_lite_risk_budget_index_only",
    }:
        return ml_external_soft_lite_risk_budget_position(frame, params), pd.Series("EXTERNAL_SOFT_LITE_RISK_BUDGET", index=frame.index)  # type: ignore[arg-type]
    if family in {
        "ml_external_soft_lite_early_stress",
        "ml_external_soft_lite_early_stress_index_only",
    }:
        return ml_external_soft_lite_early_stress_position(frame, params), pd.Series("EXTERNAL_SOFT_LITE_EARLY_STRESS", index=frame.index)  # type: ignore[arg-type]
    if family == "ml_big_up_plus_115":
        return plus_exposure_position(frame, params, cap=1.15), pd.Series("PLUS", index=frame.index)  # type: ignore[arg-type]
    if family == "ml_big_up_plus_120":
        return plus_exposure_position(frame, params, cap=1.20), pd.Series("PLUS", index=frame.index)  # type: ignore[arg-type]
    raise ValueError(f"Unknown strategy family: {family}")
