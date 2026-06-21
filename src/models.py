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
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def make_mlp_classifier(
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    feature_top_k: int | None = None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if feature_top_k is not None:
        steps.append(("selector", SelectKBest(score_func=f_classif, k=feature_top_k)))
    steps.extend(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size="auto",
                    learning_rate="adaptive",
                    learning_rate_init=learning_rate_init,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    max_iter=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def make_mlp_regressor(
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    feature_top_k: int | None = None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = []
    if feature_top_k is not None:
        steps.append(("selector", SelectKBest(score_func=f_regression, k=feature_top_k)))
    steps.extend(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    batch_size="auto",
                    learning_rate="adaptive",
                    learning_rate_init=learning_rate_init,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    max_iter=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def _mlp_spec(
    name: str,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    feature_top_k: int | None = None,
) -> ModelSpec:
    return ModelSpec(
        name,
        make_mlp_regressor(hidden_layer_sizes, alpha, learning_rate_init, feature_top_k),
        make_mlp_classifier(hidden_layer_sizes, alpha, learning_rate_init, feature_top_k),
        make_mlp_classifier(hidden_layer_sizes, alpha, learning_rate_init, feature_top_k),
    )


def _linear_logistic_spec(
    name: str,
    ridge_alpha: float,
    logistic_c: float,
    class_weight: str | None = "balanced",
) -> ModelSpec:
    return ModelSpec(
        name,
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE)),
            ]
        ),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=logistic_c,
                        solver="lbfgs",
                        class_weight=class_weight,
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=logistic_c,
                        solver="lbfgs",
                        class_weight=class_weight,
                        max_iter=3000,
                        random_state=RANDOM_STATE + 1,
                    ),
                ),
            ]
        ),
    )


def _ridge_classifier_spec(name: str, ridge_alpha: float) -> ModelSpec:
    return ModelSpec(
        name,
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE)),
            ]
        ),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeClassifier(alpha=ridge_alpha, class_weight="balanced")),
            ]
        ),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeClassifier(alpha=ridge_alpha, class_weight="balanced")),
            ]
        ),
    )


def _elastic_net_spec(name: str, alpha: float, l1_ratio: float) -> ModelSpec:
    classifier = LogisticRegression(
        C=0.5,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    trade_classifier = LogisticRegression(
        C=0.5,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE + 1,
    )
    return ModelSpec(
        name,
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        Pipeline([("scaler", StandardScaler()), ("model", classifier)]),
        Pipeline([("scaler", StandardScaler()), ("model", trade_classifier)]),
    )


def build_mlp_feature_screen_specs(max_features: int | None = None) -> list[ModelSpec]:
    return [
        _mlp_spec(f"mlp_small_fs{top_k}", (64, 32), 1e-3, 1e-3, feature_top_k=top_k)
        for top_k in (20, 30, 50, 80)
        if max_features is None or top_k <= max_features
    ]


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
        _mlp_spec("mlp_small", (64, 32), 1e-3, 1e-3),
        _mlp_spec("mlp_tiny", (32,), 3e-3, 5e-4),
        _mlp_spec("mlp_small_fs30", (64, 32), 1e-3, 1e-3, feature_top_k=30),
        _linear_logistic_spec("LogisticRegression", ridge_alpha=1.0, logistic_c=1.0),
        _linear_logistic_spec("LogisticRegressionStrongL2", ridge_alpha=10.0, logistic_c=0.2),
        _ridge_classifier_spec("RidgeClassifier", ridge_alpha=10.0),
        _elastic_net_spec("ElasticNet", alpha=0.1, l1_ratio=0.2),
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
