from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.config import COST_RATE, OPPORTUNITY_COST_BUFFER


StrategyFamily = Literal[
    "ml_big_up_index_enhancement",
    "ml_bull_full_participation_enhancement",
    "ml_conservative_full_participation_enhancement",
    "ml_ultra_conservative_full_participation_enhancement",
    "ml_direct_signal_timing",
    "ml_big_up_plus_115",
    "ml_big_up_plus_120",
]


@dataclass(frozen=True)
class StrategyParams:
    """策略参数对象，包含信号来源、阈值与仓位控制约束。"""

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
    decision_target_label: str = "big_up_label"
    opportunity_cost_buffer: float = OPPORTUNITY_COST_BUFFER
    min_strong_up_position: float = 0.95
    # 每日仓位最大变化幅度（仓位平滑）。1.0 表示不限制；典型取值 0.2/0.3。
    max_position_change: float = 1.0

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
    """合并特征上下文与指定模型/周期的预测结果，供策略层统一使用。"""
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
    """仅当目标仓位变化足够大时才调仓，避免高换手小幅抖动。"""
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
    """在回到满仓时允许立即复位，其余情况下沿用 no-trade band。"""
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


def apply_position_smoothing(
    position: pd.Series,
    max_position_change: float,
    initial_position: float = 1.0,
) -> pd.Series:
    """限制每日仓位的最大变化幅度，平滑模型短期噪声造成的过度交易与回撤。

    逻辑（与提示词伪代码一致）：
        previous_position = positions[i - 1]
        raw_position      = positions[i]
        if raw > previous + max_change:  positions[i] = previous + max_change
        elif raw < previous - max_change: positions[i] = previous - max_change
        else:                             positions[i] = raw

    这样仓位每天最多只朝目标方向移动 max_position_change（例如 0.2/0.3），
    避免“今天满仓、明天大幅砍仓、后天又满仓”这种由短期信号抖动驱动的高换手。
    当 max_position_change >= 1.0 时不产生任何约束（用于关闭该机制）。
    注意：本函数只依赖 t 日及之前已实现的仓位，不使用任何未来信息，无数据泄漏。
    """
    if max_position_change >= 1.0:
        # 上限不小于满仓跨度时，平滑器不会改变任何值，直接返回原序列。
        return position.astype(float)
    values: list[float] = []
    previous = float(initial_position)
    for value in position.fillna(initial_position).astype(float):
        target = float(value)
        if target > previous + max_position_change:
            current = previous + max_position_change
        elif target < previous - max_position_change:
            current = previous - max_position_change
        else:
            current = target
        # 平滑后的仓位严格保持在 [0, 1]，确保无杠杆约束不被破坏。
        current = min(1.0, max(0.0, current))
        values.append(current)
        previous = current
    return pd.Series(values, index=position.index, dtype=float)


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


def _strong_up_trend_flags(frame: pd.DataFrame) -> pd.Series:
    """识别强上涨阶段，避免策略在牛市中轻易低于高仓位。"""
    trend_score = frame.get("trend_regime_score", pd.Series(0.5, index=frame.index)).fillna(0.5)
    ret_20 = frame.get("ret_20", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ret_60 = frame.get("ret_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ma_ratio_20 = frame.get("ma_ratio_20", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ma_ratio_60 = frame.get("ma_ratio_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    return (
        (trend_score >= 0.62)
        & (ret_20 > 0.015)
        & (ret_60 > 0.03)
        & (ma_ratio_20 > 0)
        & (ma_ratio_60 > 0)
    ).fillna(False)


def _decision_signals(frame: pd.DataFrame, params: StrategyParams) -> dict[str, pd.Series]:
    """按照实验选择的目标标签提取减仓价值与上涨参与信号。"""
    label = params.decision_target_label
    label_to_prob = {
        "big_up_label": frame.get("pred_big_up_label_prob", frame.get("pred_big_up_prob", pd.Series(0.5, index=frame.index))),
        "big_down_label": frame.get("pred_big_down_label_prob", frame.get("pred_big_down_prob", pd.Series(0.5, index=frame.index))),
        "avoid_loss_label": frame.get("pred_avoid_loss_label_prob", pd.Series(0.5, index=frame.index)),
        "reduce_position_worth_label": frame.get("pred_reduce_position_worth_label_prob", pd.Series(0.5, index=frame.index)),
    }
    selected_prob = label_to_prob.get(label, label_to_prob["big_up_label"]).fillna(0.5)
    pred_future_drawdown = frame.get("pred_future_drawdown", pd.Series(0.0, index=frame.index)).fillna(0.0)
    pred_ret = frame.get("pred_ret", pd.Series(0.0, index=frame.index)).fillna(0.0)
    pred_big_up = frame.get("pred_big_up_prob", pd.Series(0.5, index=frame.index)).fillna(0.5)
    pred_big_down = frame.get("pred_big_down_prob", pd.Series(0.5, index=frame.index)).fillna(0.5)
    up_prob = frame.get("pred_up_prob", pd.Series(0.5, index=frame.index)).fillna(0.5)
    benefit_of_reducing = (-pred_future_drawdown).clip(lower=0.0) + pred_big_down.clip(lower=0.0) * 0.01
    # 机会成本门槛向上调整，确保策略不会因为轻微信号噪声就减仓。
    benefit_threshold = (
        COST_RATE
        + params.opportunity_cost_buffer
        + pred_big_up.clip(lower=0.0) * 0.006
        + up_prob.clip(lower=0.0) * 0.003
        + pred_ret.clip(lower=0.0) * 1.2
    )
    reduce_is_worth = benefit_of_reducing > benefit_threshold
    return {
        "selected_prob": selected_prob,
        "pred_future_drawdown": pred_future_drawdown,
        "pred_ret": pred_ret,
        "pred_big_up": pred_big_up,
        "pred_big_down": pred_big_down,
        "pred_up_prob": up_prob,
        "reduce_is_worth": reduce_is_worth.fillna(False),
    }


def _bullish_protection_flags(frame: pd.DataFrame, params: StrategyParams) -> pd.Series:
    """识别应当优先保留高仓位的强趋势上涨场景。"""
    signals = _decision_signals(frame, params)
    strong_up_trend = _strong_up_trend_flags(frame)
    pred_ret = signals["pred_ret"]
    pred_big_up = signals["pred_big_up"]
    pred_up_prob = signals["pred_up_prob"]
    tail = frame.get("pred_tail_score", pd.Series(0.0, index=frame.index)).fillna(0.0)
    supportive_signal = (
        (pred_big_up >= max(0.50, params.big_up_prob_threshold - 0.03))
        | (pred_up_prob >= 0.52)
        | (pred_ret > -0.0005)
        | (tail >= max(-0.005, params.tail_score_threshold - 0.02))
    )
    return (strong_up_trend & supportive_signal).fillna(False)


def big_up_index_enhancement_position(frame: pd.DataFrame, params: StrategyParams) -> pd.Series:
    """指数增强默认接近满仓，只有减仓显著更划算时才降低仓位。"""
    signals = _decision_signals(frame, params)
    big_up = signals["pred_big_up"]
    big_down = signals["pred_big_down"]
    pred_ret = signals["pred_ret"]
    future_drawdown = signals["pred_future_drawdown"]
    selected_prob = signals["selected_prob"]
    reduce_is_worth = signals["reduce_is_worth"]
    tail = frame["pred_tail_score"].fillna(0.0)
    weak_trend, risk, severe_risk = _risk_flags(frame, params)
    strong_up_trend = _strong_up_trend_flags(frame)
    bullish_protect = _bullish_protection_flags(frame, params)

    positive = (
        (big_up >= params.big_up_prob_threshold)
        | (tail >= params.tail_score_threshold)
        | (selected_prob >= params.big_up_prob_threshold)
        | (pred_ret > params.opportunity_cost_buffer)
    )
    downside = (
        (big_down >= params.big_down_prob_threshold)
        | (tail <= -params.tail_score_threshold)
        | ((selected_prob >= params.big_down_prob_threshold) & reduce_is_worth)
        | ((future_drawdown <= -(params.opportunity_cost_buffer + COST_RATE)) & reduce_is_worth)
    )
    strong_positive = positive & ~weak_trend & ~risk & ~severe_risk
    mild_risk = (weak_trend | risk | downside) & reduce_is_worth & ~severe_risk & ~bullish_protect
    defensive = (downside & (weak_trend | risk) & reduce_is_worth) & ~severe_risk & ~bullish_protect

    raw = pd.Series(0.98, index=frame.index, dtype=float)
    raw = raw.mask(strong_positive, 1.00)
    raw = raw.mask(mild_risk, params.mild_cut_exposure)
    raw = raw.mask(defensive, params.defensive_cut_exposure)
    raw = raw.mask(severe_risk, params.severe_defensive_exposure)
    # 强上涨期默认保留高仓位，避免因为轻微信号噪声而错过主升段。
    raw = raw.mask(strong_up_trend & ~severe_risk, np.maximum(raw, max(params.min_strong_up_position, 0.985)))
    raw = raw.mask(bullish_protect & ~severe_risk, np.maximum(raw, 0.99))
    raw = raw.clip(lower=0.0, upper=1.0)
    # 先用 no-trade band 抑制小幅抖动，再用仓位平滑限制单日最大变化幅度。
    banded = apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0)
    return apply_position_smoothing(banded, params.max_position_change, initial_position=1.0)


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
    """保守增强策略偏向满仓，仅在负面证据充分且值得减仓时才动作。"""
    big_down = frame["pred_big_down_prob"].fillna(0.5)
    tail = frame["pred_tail_score"].fillna(0.0)
    pred_ret = frame["pred_ret"].fillna(0.0)
    up_prob = frame["pred_up_prob"].fillna(0.5)
    reduce_is_worth = _decision_signals(frame, params)["reduce_is_worth"]
    strong_up_trend = _strong_up_trend_flags(frame)
    bullish_protect = _bullish_protection_flags(frame, params)
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
    mild_cut = (evidence_count >= 2) & reduce_is_worth & ~bullish_protect
    risk_cut = ml_negative & weak_trend & high_risk & reduce_is_worth & ~bullish_protect
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
    raw = raw.mask(strong_up_trend & ~severe_cut, np.maximum(raw, max(params.min_strong_up_position, 0.985)))
    raw = raw.mask(bullish_protect & ~severe_cut, np.maximum(raw, 0.99))
    raw = raw.clip(lower=0.92, upper=1.0)
    banded = apply_no_trade_band(raw, params.no_trade_band, initial_position=1.0)
    return apply_position_smoothing(banded, params.max_position_change, initial_position=1.0)


def ultra_conservative_full_participation_position(frame: pd.DataFrame, params: StrategyParams) -> pd.Series:
    """超保守增强策略仍以高仓位为默认值，只在高把握风险阶段减仓。"""
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
    reduce_is_worth = _decision_signals(frame, params)["reduce_is_worth"]
    strong_up_trend = _strong_up_trend_flags(frame)
    bullish_protect = _bullish_protection_flags(frame, params)

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

    mild_cut = (weak_signal_count >= 4) & reduce_is_worth & ~bullish_protect
    defensive_cut = ml_negative & weak_trend & high_risk & reduce_is_worth & ~bullish_protect
    severe_cut = strong_ml_negative & weak_trend & high_risk & stress & reduce_is_worth

    raw = pd.Series(1.0, index=frame.index, dtype=float)
    raw = raw.mask(mild_cut, params.mild_cut_exposure)
    raw = raw.mask(defensive_cut, params.defensive_cut_exposure)
    raw = raw.mask(severe_cut, params.severe_defensive_exposure)
    raw = raw.mask(strong_up_trend & ~severe_cut, np.maximum(raw, max(params.min_strong_up_position, 0.99)))
    raw = raw.mask(bullish_protect & ~severe_cut, np.maximum(raw, 0.995))
    raw = raw.clip(lower=0.94, upper=1.0)
    banded = apply_no_trade_band_with_full_reset(raw, params.no_trade_band, initial_position=1.0)
    return apply_position_smoothing(banded, params.max_position_change, initial_position=1.0)


def bull_full_participation_position(frame: pd.DataFrame, params: StrategyParams) -> pd.Series:
    """牛市优先增强策略：默认满仓，仅在极端负面证据下轻微减仓。"""
    signals = _decision_signals(frame, params)
    big_down = signals["pred_big_down"]
    pred_ret = signals["pred_ret"]
    up_prob = signals["pred_up_prob"]
    reduce_is_worth = signals["reduce_is_worth"]
    tail = frame["pred_tail_score"].fillna(0.0)
    weak_trend, risk, severe_risk = _risk_flags(frame, params)
    bullish_protect = _bullish_protection_flags(frame, params)
    drawdown_60 = frame.get("drawdown_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    vol_ratio = frame.get("volatility_ratio_20_60", pd.Series(1.0, index=frame.index)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    extreme_negative = (
        (big_down >= params.big_down_prob_threshold + 0.05)
        | (tail <= -abs(params.tail_score_threshold) - 0.03)
        | (pred_ret < -0.006)
        | ((up_prob < 0.42) & weak_trend)
    )
    stress_negative = extreme_negative & risk & reduce_is_worth
    severe_negative = stress_negative & (
        severe_risk
        | (drawdown_60 < -0.16)
        | (vol_ratio > params.high_volatility_multiplier + 0.20)
    )

    raw = pd.Series(1.0, index=frame.index, dtype=float)
    raw = raw.mask(stress_negative & ~bullish_protect, np.maximum(params.defensive_cut_exposure, 0.985))
    raw = raw.mask(severe_negative, np.maximum(params.severe_defensive_exposure, 0.96))
    raw = raw.mask(bullish_protect & ~severe_negative, 1.0)
    raw = raw.clip(lower=0.96, upper=1.0)
    banded = apply_no_trade_band_with_full_reset(raw, max(params.no_trade_band, 0.03), initial_position=1.0)
    return apply_position_smoothing(banded, params.max_position_change, initial_position=1.0)


def build_strategy_positions(frame: pd.DataFrame, params: StrategyParams, family: StrategyFamily) -> tuple[pd.Series, pd.Series]:
    if family == "ml_big_up_index_enhancement":
        return big_up_index_enhancement_position(frame, params), pd.Series("ADJUST", index=frame.index)
    if family == "ml_bull_full_participation_enhancement":
        return bull_full_participation_position(frame, params), pd.Series("BULL_FULL", index=frame.index)
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
