from __future__ import annotations

import numpy as np
import pandas as pd


def fit_preprocessor(train: pd.DataFrame, feature_cols: list[str]) -> dict[str, pd.Series]:
    x = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    lower = x.quantile(0.01)
    upper = x.quantile(0.99)
    med = x.clip(lower=lower, upper=upper, axis=1).median().fillna(0.0)
    lower = lower.fillna(med)
    upper = upper.fillna(med)
    return {"lower": lower, "upper": upper, "median": med}


def transform_features(df: pd.DataFrame, feature_cols: list[str], prep: dict[str, pd.Series]) -> pd.DataFrame:
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x = x.clip(lower=prep["lower"], upper=prep["upper"], axis=1)
    x = x.fillna(prep["median"])
    return x.astype(float)
