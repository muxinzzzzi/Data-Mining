from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS, OPPORTUNITY_COST_BUFFER, TRADE_THRESHOLD
from .utils import historical_target_quantiles


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """为每个预测周期构建严格无泄漏的收益、尾部风险与仓位决策标签。"""
    out = df.copy()
    out["trade_date"] = out["date"].shift(-1)
    out["fwd_ret_1d"] = out["close"].shift(-1) / out["close"] - 1

    for h in HORIZONS:
        ret = out["close"].shift(-h) / out["close"] - 1
        target_end_date = out["date"].shift(-h)
        future_path_returns = pd.concat(
            [(out["close"].shift(-step) / out["close"] - 1).rename(step) for step in range(1, h + 1)],
            axis=1,
        )
        future_drawdown = future_path_returns.min(axis=1)

        big_up_threshold = historical_target_quantiles(out["date"], target_end_date, ret, 0.70)
        big_down_threshold = historical_target_quantiles(out["date"], target_end_date, ret, 0.30)
        clean_up_threshold = historical_target_quantiles(out["date"], target_end_date, ret, 0.60)
        clean_down_threshold = historical_target_quantiles(out["date"], target_end_date, ret, 0.40)

        clean_direction = pd.Series(np.nan, index=out.index, dtype=float)
        clean_direction = clean_direction.mask(ret > clean_up_threshold, 1.0)
        clean_direction = clean_direction.mask(ret < clean_down_threshold, 0.0)

        out[f"target_ret_{h}d"] = ret
        out[f"target_up_{h}d"] = np.where(ret.notna(), (ret > 0).astype(float), np.nan)
        out[f"target_trade_{h}d"] = np.where(ret.notna(), (ret > TRADE_THRESHOLD).astype(float), np.nan)
        out[f"target_big_up_{h}d"] = np.where(
            ret.notna() & big_up_threshold.notna(),
            (ret > big_up_threshold).astype(float),
            np.nan,
        )
        out[f"target_big_down_{h}d"] = np.where(
            ret.notna() & big_down_threshold.notna(),
            (ret < big_down_threshold).astype(float),
            np.nan,
        )
        out[f"target_clean_direction_{h}d"] = clean_direction
        out[f"target_future_drawdown_{h}d"] = future_drawdown
        out[f"target_big_up_label_{h}d"] = out[f"target_big_up_{h}d"]
        out[f"target_big_down_label_{h}d"] = out[f"target_big_down_{h}d"]
        out[f"target_avoid_loss_label_{h}d"] = np.where(
            ret.notna() & future_drawdown.notna(),
            ((ret <= -TRADE_THRESHOLD) | (future_drawdown <= -TRADE_THRESHOLD)).astype(float),
            np.nan,
        )
        out[f"target_reduce_position_worth_label_{h}d"] = np.where(
            ret.notna() & future_drawdown.notna(),
            (
                (future_drawdown <= -(TRADE_THRESHOLD + OPPORTUNITY_COST_BUFFER))
                & (ret <= OPPORTUNITY_COST_BUFFER)
            ).astype(float),
            np.nan,
        )
        out[f"target_big_up_threshold_{h}d"] = big_up_threshold
        out[f"target_big_down_threshold_{h}d"] = big_down_threshold
        out[f"target_clean_up_threshold_{h}d"] = clean_up_threshold
        out[f"target_clean_down_threshold_{h}d"] = clean_down_threshold
        out[f"target_start_date_{h}d"] = out["date"].shift(-1)
        out[f"target_end_date_{h}d"] = target_end_date

    return out
