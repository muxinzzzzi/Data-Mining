from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from .config import RANDOM_STATE


@dataclass(frozen=True)
class ModelSpec:
    name: str
    reg: Any
    cls: Any
    trade_cls: Any
    big_up_cls: Any | None = None
    big_down_cls: Any | None = None
    clean_direction_cls: Any | None = None


def clone_model(model: Any) -> Any:
    from sklearn.base import clone

    return clone(model)


def fit_binary_probability(model: Any, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame) -> np.ndarray:
    y = y_train.astype(int)
    if len(y) == 0:
        return np.full(len(x_test), 0.5)
    if y.nunique() < 2:
        return np.full(len(x_test), float(y.iloc[0]))

    fitted = clone_model(model)
    fitted.fit(x_train, y)
    if hasattr(fitted, "predict_proba"):
        proba = fitted.predict_proba(x_test)
        classes = list(fitted.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(x_test))
    raw = fitted.decision_function(x_test)
    return 1 / (1 + np.exp(-raw))


def _hist_gb_spec() -> ModelSpec:
    return ModelSpec(
        "HistGradientBoosting",
        HistGradientBoostingRegressor(
            learning_rate=0.04,
            max_iter=80,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
        HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=80,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
        HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=80,
            max_leaf_nodes=15,
            min_samples_leaf=25,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=RANDOM_STATE + 1,
        ),
    )


def _random_forest_spec() -> ModelSpec:
    return ModelSpec(
        "RandomForest",
        RandomForestRegressor(
            n_estimators=40,
            max_depth=6,
            min_samples_leaf=18,
            max_features="sqrt",
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=40,
            max_depth=6,
            min_samples_leaf=18,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=40,
            max_depth=6,
            min_samples_leaf=18,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE + 1,
            n_jobs=1,
        ),
    )


def _extra_trees_spec() -> ModelSpec:
    return ModelSpec(
        "ExtraTrees",
        ExtraTreesRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=16,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        ExtraTreesClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=16,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        ExtraTreesClassifier(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=16,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE + 1,
            n_jobs=1,
        ),
    )


def _gradient_boosting_spec() -> ModelSpec:
    return ModelSpec(
        "GradientBoosting",
        GradientBoostingRegressor(
            n_estimators=60,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=18,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=18,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=18,
            subsample=0.8,
            random_state=RANDOM_STATE + 1,
        ),
    )


def _extra_trees_deep_spec() -> ModelSpec:
    return ModelSpec(
        "ExtraTreesDeep",
        ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=8,
            max_features=0.5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=8,
            max_features=0.5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=8,
            max_features=0.5,
            class_weight="balanced",
            random_state=RANDOM_STATE + 1,
            n_jobs=1,
        ),
    )


def _random_forest_deep_spec() -> ModelSpec:
    return ModelSpec(
        "RandomForestDeep",
        RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            max_features=0.5,
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            max_features=0.5,
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            max_features=0.5,
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE + 1,
            n_jobs=1,
        ),
    )


def _xgb_spec(name: str, n_estimators: int, learning_rate: float, max_depth: int, reg_lambda: float, reg_alpha: float):
    from xgboost import XGBClassifier, XGBRegressor

    kwargs = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "verbosity": 0,
    }
    cls_kwargs = {
        **kwargs,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    }
    return ModelSpec(
        name,
        XGBRegressor(**kwargs, objective="reg:squarederror"),
        XGBClassifier(**cls_kwargs),
        XGBClassifier(**{**cls_kwargs, "random_state": RANDOM_STATE + 1}),
    )


def _lgbm_spec(name: str, n_estimators: int, learning_rate: float, num_leaves: int, reg_lambda: float, reg_alpha: float):
    from lightgbm import LGBMClassifier, LGBMRegressor

    kwargs = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": 3,
        "min_child_samples": 20,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "verbose": -1,
    }
    return ModelSpec(
        name,
        LGBMRegressor(**kwargs),
        LGBMClassifier(**kwargs),
        LGBMClassifier(**{**kwargs, "random_state": RANDOM_STATE + 1}),
    )


def build_model_specs() -> tuple[list[ModelSpec], list[dict[str, str]]]:
    specs = [
        _hist_gb_spec(),
        _random_forest_spec(),
        _extra_trees_spec(),
        _gradient_boosting_spec(),
        _extra_trees_deep_spec(),
        _random_forest_deep_spec(),
    ]

    skipped: list[dict[str, str]] = []
    try:
        specs.append(_xgb_spec("XGBoost", 200, 0.03, 3, 1.0, 0.0))
        specs.append(_xgb_spec("XGBoostRegularized", 300, 0.02, 3, 3.0, 0.2))
    except Exception as exc:
        skipped.append({"model_name": "XGBoost/XGBoostRegularized", "reason": f"xgboost unavailable: {exc.__class__.__name__}"})

    try:
        specs.append(_lgbm_spec("LightGBM", 200, 0.03, 15, 1.0, 0.0))
        specs.append(_lgbm_spec("LightGBMRegularized", 300, 0.02, 7, 3.0, 0.2))
    except Exception as exc:
        skipped.append({"model_name": "LightGBM/LightGBMRegularized", "reason": f"lightgbm unavailable: {exc.__class__.__name__}"})

    return specs, skipped
