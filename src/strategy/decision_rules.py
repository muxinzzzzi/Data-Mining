from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


StrategyFamily = Literal[
    "ml_big_up_index_enhancement",
    "ml_conservative_full_participation_enhancement",
    "ml_ultra_conservative_full_participation_enhancement",
    "ml_direct_signal_timing",
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


FEATURE_CONTEXT_COLS = [
    "date",
    "ret_5",
    "ret_20",
    "ret_60",
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
    "crash_risk_score",
    "m5_available",
]


def prepare_strategy_frame(feature_df: pd.DataFrame, predictions: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    pred = predictions.loc[
        (predictions["model_name"] == params.model_name) & (predictions["horizon"] == params.horizon)
    ].copy()
    if pred.empty:
        raise ValueError(f"No predictions for {params.model_name}, horizon={params.horizon}")
    context_cols = [col for col in FEATURE_CONTEXT_COLS if col in feature_df.columns]
    context = feature_df[context_cols].copy()
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


def build_strategy_positions(frame: pd.DataFrame, params: StrategyParams, family: StrategyFamily) -> tuple[pd.Series, pd.Series]:
    if family == "ml_big_up_index_enhancement":
        return big_up_index_enhancement_position(frame, params), pd.Series("ADJUST", index=frame.index)
    if family == "ml_conservative_full_participation_enhancement":
        return conservative_full_participation_position(frame, params), pd.Series("CONSERVATIVE", index=frame.index)
    if family == "ml_ultra_conservative_full_participation_enhancement":
        return ultra_conservative_full_participation_position(frame, params), pd.Series("ULTRA_CONSERVATIVE", index=frame.index)
    if family == "ml_direct_signal_timing":
        return direct_signal_timing_position(frame, params)
    if family == "ml_big_up_plus_115":
        return plus_exposure_position(frame, params, cap=1.15), pd.Series("PLUS", index=frame.index)
    if family == "ml_big_up_plus_120":
        return plus_exposure_position(frame, params, cap=1.20), pd.Series("PLUS", index=frame.index)
    raise ValueError(f"Unknown strategy family: {family}")
