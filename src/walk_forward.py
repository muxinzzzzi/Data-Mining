from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    FEATURE_TOP_K_CANDIDATES,
    HORIZONS,
    MIN_TASK_SAMPLES,
    MIN_TRAIN_SAMPLES,
    VALID_END,
    VALID_RETRAIN_EVERY,
    VALID_START,
)
from .feature_selection import feature_top_k_label, select_task_feature_columns
from .metrics import compute_model_metrics, score_validation_metrics_for_feature_selection
from .models import ModelSpec, clone_model, fit_binary_probability
from .preprocessing import fit_preprocessor, transform_features


@dataclass
class TaskBundle:
    selected_features: list[str]
    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    train_samples: int


def _predict_regression_from_bundle(model: Any, bundle: TaskBundle, default: float = 0.0) -> np.ndarray:
    """在样本不足时安全退化为训练均值，避免单个任务中断整个 walk-forward。"""
    if bundle.train_samples < MIN_TASK_SAMPLES:
        if bundle.train_samples > 0:
            return np.full(len(bundle.x_test), float(bundle.y_train.mean()))
        return np.full(len(bundle.x_test), default)
    fitted = clone_model(model)
    fitted.fit(bundle.x_train, bundle.y_train.astype(float))
    return fitted.predict(bundle.x_test)


def _make_task_bundle(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    top_k: int | None,
    fallback_target_col: str,
) -> TaskBundle:
    selected = select_task_feature_columns(train, feature_cols, target_col, top_k, fallback_target_col)
    valid = train[target_col].replace([np.inf, -np.inf], np.nan).notna() if target_col in train.columns else pd.Series(False, index=train.index)
    task_train = train.loc[valid].copy()
    if task_train.empty:
        task_train = train.loc[train[fallback_target_col].notna()].copy()
        y_train = pd.Series(dtype=float)
    else:
        y_train = task_train[target_col].astype(float)
    prep = fit_preprocessor(task_train, selected)
    x_train = transform_features(task_train, selected, prep).loc[y_train.index] if len(y_train) else pd.DataFrame(columns=selected)
    x_test = transform_features(test, selected, prep)
    return TaskBundle(selected, x_train, y_train, x_test, int(len(y_train)))


def _predict_binary_from_bundle(model: Any, bundle: TaskBundle, default: float = 0.5) -> np.ndarray:
    if bundle.train_samples < MIN_TASK_SAMPLES:
        if bundle.train_samples > 0:
            return np.full(len(bundle.x_test), float(bundle.y_train.mean()))
        return np.full(len(bundle.x_test), default)
    return fit_binary_probability(model, bundle.x_train, bundle.y_train, bundle.x_test)


def walk_forward_predictions(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_specs: list[ModelSpec],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    retrain_every: int,
    min_train_samples: int = MIN_TRAIN_SAMPLES,
    label: str = "test",
    feature_top_k: int | None = None,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """基于固定测试窗口执行无泄漏 walk-forward 预测，并输出多标签结果。"""
    active_horizons = horizons or HORIZONS
    test_dates = (
        df.loc[(df["date"] >= period_start) & (df["date"] <= period_end), "date"]
        .dropna()
        .sort_values()
        .drop_duplicates()
        .tolist()
    )
    all_preds: list[pd.DataFrame] = []

    for h in active_horizons:
        ret_col = f"target_ret_{h}d"
        up_col = f"target_up_{h}d"
        trade_col = f"target_trade_{h}d"
        big_up_col = f"target_big_up_{h}d"
        big_down_col = f"target_big_down_{h}d"
        clean_col = f"target_clean_direction_{h}d"
        future_drawdown_col = f"target_future_drawdown_{h}d"
        big_up_label_col = f"target_big_up_label_{h}d"
        big_down_label_col = f"target_big_down_label_{h}d"
        avoid_loss_label_col = f"target_avoid_loss_label_{h}d"
        reduce_position_worth_col = f"target_reduce_position_worth_label_{h}d"
        end_col = f"target_end_date_{h}d"
        start_col = f"target_start_date_{h}d"

        for chunk_idx in range(0, len(test_dates), retrain_every):
            chunk_dates = test_dates[chunk_idx : chunk_idx + retrain_every]
            if not chunk_dates:
                continue
            chunk_start = pd.Timestamp(chunk_dates[0])
            train_mask = (df[end_col] < chunk_start) & df[ret_col].notna()
            test_mask = df["date"].isin(chunk_dates)
            train = df.loc[train_mask].copy()
            test = df.loc[test_mask].copy()
            if len(train) < min_train_samples or test.empty:
                continue

            reg_bundle = _make_task_bundle(train, test, feature_cols, ret_col, feature_top_k, ret_col)
            up_bundle = _make_task_bundle(train, test, feature_cols, up_col, feature_top_k, ret_col)
            trade_bundle = _make_task_bundle(train, test, feature_cols, trade_col, feature_top_k, ret_col)
            big_up_bundle = _make_task_bundle(train, test, feature_cols, big_up_col, feature_top_k, ret_col)
            big_down_bundle = _make_task_bundle(train, test, feature_cols, big_down_col, feature_top_k, ret_col)
            clean_bundle = _make_task_bundle(train, test, feature_cols, clean_col, feature_top_k, ret_col)
            future_drawdown_bundle = _make_task_bundle(train, test, feature_cols, future_drawdown_col, feature_top_k, ret_col)
            big_up_label_bundle = _make_task_bundle(train, test, feature_cols, big_up_label_col, feature_top_k, ret_col)
            big_down_label_bundle = _make_task_bundle(train, test, feature_cols, big_down_label_col, feature_top_k, ret_col)
            avoid_loss_label_bundle = _make_task_bundle(train, test, feature_cols, avoid_loss_label_col, feature_top_k, ret_col)
            reduce_position_worth_bundle = _make_task_bundle(
                train,
                test,
                feature_cols,
                reduce_position_worth_col,
                feature_top_k,
                ret_col,
            )

            base_cols = [
                "date",
                "trade_date",
                start_col,
                end_col,
                "close",
                "fwd_ret_1d",
                ret_col,
                up_col,
                trade_col,
                big_up_col,
                big_down_col,
                clean_col,
                future_drawdown_col,
                big_up_label_col,
                big_down_label_col,
                avoid_loss_label_col,
                reduce_position_worth_col,
            ]

            for spec in model_specs:
                try:
                    pred_ret = _predict_regression_from_bundle(spec.reg, reg_bundle)
                    pred_up_prob = _predict_binary_from_bundle(spec.cls, up_bundle)
                    pred_trade_prob = _predict_binary_from_bundle(spec.trade_cls, trade_bundle)
                    big_up_model = spec.big_up_cls if spec.big_up_cls is not None else spec.trade_cls
                    big_down_model = spec.big_down_cls if spec.big_down_cls is not None else spec.trade_cls
                    clean_model = spec.clean_direction_cls if spec.clean_direction_cls is not None else spec.cls
                    pred_big_up_prob = _predict_binary_from_bundle(big_up_model, big_up_bundle)
                    pred_big_down_prob = _predict_binary_from_bundle(big_down_model, big_down_bundle)
                    pred_clean_prob = _predict_binary_from_bundle(clean_model, clean_bundle)
                    pred_future_drawdown = _predict_regression_from_bundle(spec.reg, future_drawdown_bundle)
                    pred_big_up_label_prob = _predict_binary_from_bundle(big_up_model, big_up_label_bundle)
                    pred_big_down_label_prob = _predict_binary_from_bundle(big_down_model, big_down_label_bundle)
                    pred_avoid_loss_label_prob = _predict_binary_from_bundle(spec.trade_cls, avoid_loss_label_bundle)
                    pred_reduce_position_worth_prob = _predict_binary_from_bundle(
                        spec.trade_cls,
                        reduce_position_worth_bundle,
                    )
                except Exception:
                    continue

                pred = test[base_cols].copy()
                pred = pred.rename(
                    columns={
                        start_col: "target_start_date",
                        end_col: "target_end_date",
                        ret_col: "target_ret",
                        up_col: "target_up",
                        trade_col: "target_trade",
                        big_up_col: "target_big_up",
                        big_down_col: "target_big_down",
                        clean_col: "target_clean_direction",
                        future_drawdown_col: "target_future_drawdown",
                        big_up_label_col: "target_big_up_label",
                        big_down_label_col: "target_big_down_label",
                        avoid_loss_label_col: "target_avoid_loss_label",
                        reduce_position_worth_col: "target_reduce_position_worth_label",
                    }
                )
                pred["horizon"] = h
                pred["model_name"] = spec.name
                pred["pred_ret"] = pred_ret
                pred["pred_up_prob"] = pred_up_prob
                pred["pred_trade_prob"] = pred_trade_prob
                pred["pred_big_up_prob"] = pred_big_up_prob
                pred["pred_big_down_prob"] = pred_big_down_prob
                pred["pred_tail_score"] = pred["pred_big_up_prob"] - pred["pred_big_down_prob"]
                pred["pred_clean_direction_prob"] = pred_clean_prob
                pred["pred_future_drawdown"] = pred_future_drawdown
                pred["pred_big_up_label_prob"] = pred_big_up_label_prob
                pred["pred_big_down_label_prob"] = pred_big_down_label_prob
                pred["pred_avoid_loss_label_prob"] = pred_avoid_loss_label_prob
                pred["pred_reduce_position_worth_label_prob"] = pred_reduce_position_worth_prob
                pred["pred_up"] = (pred["pred_up_prob"] >= 0.5).astype(int)
                pred["pred_trade"] = (pred["pred_trade_prob"] >= 0.5).astype(int)
                pred["pred_big_up"] = (pred["pred_big_up_prob"] >= 0.5).astype(int)
                pred["pred_big_down"] = (pred["pred_big_down_prob"] >= 0.5).astype(int)
                pred["pred_clean_direction"] = (pred["pred_clean_direction_prob"] >= 0.5).astype(int)
                pred["walk_forward_period"] = label
                pred["chunk_start"] = chunk_start
                pred["train_samples"] = len(train)
                pred["train_latest_target_end_date"] = train[end_col].max()
                pred["feature_top_k"] = feature_top_k_label(feature_top_k)
                pred["selected_feature_count"] = len(reg_bundle.selected_features)
                pred["selected_feature_count_up"] = len(up_bundle.selected_features)
                pred["selected_feature_count_trade"] = len(trade_bundle.selected_features)
                pred["selected_feature_count_big_up"] = len(big_up_bundle.selected_features)
                pred["selected_feature_count_big_down"] = len(big_down_bundle.selected_features)
                pred["selected_feature_count_clean_direction"] = len(clean_bundle.selected_features)
                pred["clean_direction_train_samples"] = clean_bundle.train_samples
                all_preds.append(pred)

    if not all_preds:
        raise RuntimeError(f"No {label} walk-forward predictions were generated.")

    out = pd.concat(all_preds, ignore_index=True)
    return out.sort_values(["date", "horizon", "model_name"]).reset_index(drop=True)


def select_validation_feature_top_k(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    model_specs: list[ModelSpec],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for top_k in FEATURE_TOP_K_CANDIDATES:
        print(f"  Validation feature top_k candidate: {feature_top_k_label(top_k)}")
        preds = walk_forward_predictions(
            feature_df,
            feature_cols,
            model_specs,
            VALID_START,
            VALID_END,
            VALID_RETRAIN_EVERY,
            min_train_samples=MIN_TRAIN_SAMPLES,
            label=f"validation_topk_{feature_top_k_label(top_k)}",
            feature_top_k=top_k,
        )
        metrics = compute_model_metrics(preds)
        score = score_validation_metrics_for_feature_selection(metrics)
        best_metric = metrics.iloc[0].to_dict() if not metrics.empty else {}
        row = {
            "top_k": top_k,
            "top_k_label": feature_top_k_label(top_k),
            "validation_score": score,
            "prediction_rows": len(preds),
            "best_metric_model": best_metric.get("model_name"),
            "best_metric_horizon": best_metric.get("horizon"),
            "best_metric_AUC": best_metric.get("AUC"),
            "best_metric_clean_direction_AUC": best_metric.get("clean_direction_AUC"),
            "best_metric_trade_AUC": best_metric.get("trade_AUC"),
            "best_metric_big_up_AUC": best_metric.get("big_up_AUC"),
            "best_metric_big_down_AUC": best_metric.get("big_down_AUC"),
            "best_metric_return_correlation": best_metric.get("return_correlation"),
        }
        print(
            "  Finished top_k="
            f"{row['top_k_label']}: score={row['validation_score']:.6f}, "
            f"rows={row['prediction_rows']}, best={row.get('best_metric_model')} {row.get('best_metric_horizon')}D"
        )
        rows.append(row)
        if selected is None or score > selected["validation_score"]:
            selected = {**row, "predictions": preds}

    if selected is None:
        raise RuntimeError("Validation feature top_k selection produced no candidates.")

    return {
        "selected_top_k": selected["top_k"],
        "selected_top_k_label": selected["top_k_label"],
        "selected_validation_score": selected["validation_score"],
        "candidates": rows,
        "predictions": selected["predictions"],
        "selection_rule": "Validation-only max of prediction selection_score. Each task selects features inside each walk-forward chunk using training data only.",
    }
