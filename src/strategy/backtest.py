from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import COST_RATE, INITIAL_CAPITAL
from src.strategy.decision_rules import StrategyFamily, StrategyParams, build_strategy_positions, prepare_strategy_frame


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1


def backtest_position_frame(
    frame: pd.DataFrame,
    final_position: pd.Series,
    strategy_name: str,
    signal: pd.Series | None = None,
    initial_position: float = 1.0,
) -> pd.DataFrame:
    out = frame.copy().reset_index(drop=True)
    out["strategy"] = strategy_name
    out["final_position"] = final_position.reset_index(drop=True).astype(float)
    out["signal"] = signal.reset_index(drop=True) if signal is not None else "ADJUST"
    prev_position = out["final_position"].shift(1).fillna(initial_position)
    out["turnover"] = (out["final_position"] - prev_position).abs()
    out["transaction_cost"] = out["turnover"] * COST_RATE
    out["strategy_return"] = out["final_position"] * out["fwd_ret_1d"].fillna(0.0) - out["transaction_cost"]
    out["buy_hold_return"] = out["fwd_ret_1d"].fillna(0.0)
    out["active_return"] = out["strategy_return"] - out["buy_hold_return"]
    out["equity"] = (1 + out["strategy_return"]).cumprod() * INITIAL_CAPITAL
    out["buy_hold_equity"] = (1 + out["buy_hold_return"]).cumprod() * INITIAL_CAPITAL
    out["drawdown"] = drawdown_series(out["equity"])
    out["buy_hold_drawdown"] = drawdown_series(out["buy_hold_equity"])
    return out


def build_buy_hold_frame(base_frame: pd.DataFrame) -> pd.DataFrame:
    signal = pd.Series("HOLD", index=base_frame.index)
    position = pd.Series(1.0, index=base_frame.index)
    return backtest_position_frame(base_frame, position, "buy_hold", signal=signal, initial_position=1.0)


def backtest_strategy(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    params: StrategyParams,
    family: StrategyFamily,
) -> pd.DataFrame:
    frame = prepare_strategy_frame(feature_df, predictions, params)
    position, signal = build_strategy_positions(frame, params, family)
    return backtest_position_frame(frame, position, family, signal=signal, initial_position=1.0)


def build_all_strategy_daily(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    selected_params: dict[str, StrategyParams],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    first_params = next(iter(selected_params.values()))
    base = prepare_strategy_frame(feature_df, predictions, first_params)
    frames.append(build_buy_hold_frame(base))
    for family, params in selected_params.items():
        frames.append(backtest_strategy(feature_df, predictions, params, family))  # type: ignore[arg-type]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["strategy", "date"]).reset_index(drop=True)
