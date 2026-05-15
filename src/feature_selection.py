from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MIN_TASK_FEATURE_SELECTION_SAMPLES


def feature_top_k_label(top_k: int | None) -> str:
    return "all" if top_k is None else str(top_k)


def select_ic_feature_columns(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    top_k: int | None,
) -> list[str]:
    if top_k is None or top_k >= len(feature_cols):
        return feature_cols
    valid = train[target_col].notna()
    x = train.loc[valid, feature_cols].replace([np.inf, -np.inf], np.nan)
    y = train.loc[valid, target_col].astype(float).replace([np.inf, -np.inf], np.nan)
    corr = x.corrwith(y).abs().replace([np.inf, -np.inf], np.nan).dropna()
    ranked = corr.sort_values(ascending=False)
    selected = ranked.head(top_k).index.tolist()
    if len(selected) < min(top_k, len(feature_cols)):
        selected_set = set(selected)
        selected.extend([col for col in feature_cols if col not in selected_set][: top_k - len(selected)])
    return selected


def select_task_feature_columns(
    train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    top_k: int | None,
    fallback_target_col: str,
) -> list[str]:
    valid = train[target_col].replace([np.inf, -np.inf], np.nan).notna() if target_col in train.columns else pd.Series(False, index=train.index)
    target = train.loc[valid, target_col] if target_col in train.columns else pd.Series(dtype=float)
    enough_valid = int(valid.sum()) >= MIN_TASK_FEATURE_SELECTION_SAMPLES
    enough_class_variation = target.nunique(dropna=True) >= 2 if len(target) else False
    if enough_valid and (target_col == fallback_target_col or enough_class_variation):
        return select_ic_feature_columns(train.loc[valid], feature_cols, target_col, top_k)
    return select_ic_feature_columns(train, feature_cols, fallback_target_col, top_k)
