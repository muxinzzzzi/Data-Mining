from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import HORIZONS, TEST_START
from .utils import signed_corr


def safe_auc(y_true: pd.Series, y_prob: pd.Series) -> float:
    valid = pd.concat([y_true, y_prob], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 10 or valid.iloc[:, 0].nunique() < 2:
        return np.nan
    return float(roc_auc_score(valid.iloc[:, 0].astype(int), valid.iloc[:, 1].astype(float)))


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series, prefix: str = "") -> dict[str, float]:
    valid = pd.concat([y_true, y_pred, y_prob], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    keys = {
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "F1": np.nan,
        "balanced_accuracy": np.nan,
        "AUC": np.nan,
        "logloss": np.nan,
    }
    if valid.empty:
        return {f"{prefix}{k}": v for k, v in keys.items()}

    y = valid.iloc[:, 0].astype(int)
    yhat = valid.iloc[:, 1].astype(int)
    out = {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "F1": float(f1_score(y, yhat, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)) if y.nunique() > 1 else np.nan,
        "AUC": safe_auc(y, valid.iloc[:, 2]),
        "logloss": float(log_loss(y, valid.iloc[:, 2].astype(float).clip(1e-6, 1 - 1e-6), labels=[0, 1])),
    }
    return {f"{prefix}{k}": v for k, v in out.items()}


def add_selection_score(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    if out.empty:
        out["selection_score"] = []
        return out
    out["selection_score"] = (
        0.18 * out["AUC"].fillna(0.5)
        + 0.24 * out["clean_direction_AUC"].fillna(0.5)
        + 0.16 * out["trade_AUC"].fillna(0.5)
        + 0.14 * out["big_up_AUC"].fillna(0.5)
        + 0.14 * out["big_down_AUC"].fillna(0.5)
        + 0.08 * out["directional_accuracy_from_regression"].fillna(0.5)
        + 0.06 * out["return_correlation"].clip(lower=0).fillna(0.0)
    )
    return out


def compute_model_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, horizon), grp in pred.groupby(["model_name", "horizon"], sort=True):
        reg_eval = grp.dropna(subset=["target_ret", "pred_ret"]).copy()
        if len(reg_eval) >= 2:
            mae = float(mean_absolute_error(reg_eval["target_ret"], reg_eval["pred_ret"]))
            rmse = float(math.sqrt(mean_squared_error(reg_eval["target_ret"], reg_eval["pred_ret"])))
            dir_acc = float(((reg_eval["target_ret"] > 0) == (reg_eval["pred_ret"] > 0)).mean())
            corr = signed_corr(reg_eval["target_ret"], reg_eval["pred_ret"])
        else:
            mae = rmse = dir_acc = corr = np.nan

        up = _binary_metrics(grp["target_up"], grp["pred_up"], grp["pred_up_prob"])
        trade = _binary_metrics(grp["target_trade"], grp["pred_trade"], grp["pred_trade_prob"], "trade_")
        big_up = _binary_metrics(grp["target_big_up"], grp["pred_big_up"], grp["pred_big_up_prob"], "big_up_")
        big_down = _binary_metrics(grp["target_big_down"], grp["pred_big_down"], grp["pred_big_down_prob"], "big_down_")
        clean = _binary_metrics(
            grp["target_clean_direction"],
            grp["pred_clean_direction"],
            grp["pred_clean_direction_prob"],
            "clean_direction_",
        )

        tail_eval = grp.dropna(subset=["target_ret", "pred_tail_score"]).copy()
        tail_corr = signed_corr(tail_eval["target_ret"], tail_eval["pred_tail_score"]) if len(tail_eval) >= 20 else np.nan
        tail_dir_acc = (
            float(((tail_eval["target_ret"] > 0) == (tail_eval["pred_tail_score"] > 0)).mean())
            if len(tail_eval) >= 2
            else np.nan
        )

        rows.append(
            {
                "model_name": model_name,
                "horizon": int(horizon),
                "n_eval": int(len(reg_eval)),
                "MAE": mae,
                "RMSE": rmse,
                "directional_accuracy_from_regression": dir_acc,
                "return_correlation": corr,
                **up,
                **trade,
                **big_up,
                **big_down,
                **clean,
                "tail_score_return_correlation": tail_corr,
                "tail_score_directional_accuracy": tail_dir_acc,
            }
        )

    metrics = add_selection_score(pd.DataFrame(rows))
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["selection_score", "clean_direction_AUC", "AUC", "return_correlation"],
            ascending=[False, False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return metrics


def score_validation_metrics_for_feature_selection(metrics: pd.DataFrame) -> float:
    if metrics.empty:
        return -np.inf
    scored = add_selection_score(metrics)
    return float(scored["selection_score"].max())


def select_best_prediction_model(validation_metrics: pd.DataFrame, test_metrics: pd.DataFrame | None = None) -> dict[str, Any]:
    if validation_metrics.empty:
        raise ValueError("Validation metrics are empty; cannot select best prediction model.")
    scored = add_selection_score(validation_metrics).sort_values(
        ["selection_score", "clean_direction_AUC", "AUC"],
        ascending=[False, False, False],
        na_position="last",
    )
    best = scored.iloc[0].to_dict()
    out = {
        "selection_basis": "validation_prediction_metrics_only",
        "selection_score_formula": (
            "0.18*AUC + 0.24*clean_direction_AUC + 0.16*trade_AUC + "
            "0.14*big_up_AUC + 0.14*big_down_AUC + "
            "0.08*directional_accuracy_from_regression + 0.06*max(return_correlation, 0)"
        ),
        "best_validation": best,
    }
    if test_metrics is not None and not test_metrics.empty:
        match = test_metrics.loc[
            (test_metrics["model_name"] == best["model_name"]) & (test_metrics["horizon"] == best["horizon"])
        ]
        out["matching_test"] = match.iloc[0].to_dict() if not match.empty else None
    return out


def compute_factor_effectiveness(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    label = "target_ret_5d"
    up = "target_up_5d"
    end = "target_end_date_5d"
    sample = df.loc[(df[end] < TEST_START) & df[label].notna()].copy()
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        x = sample[col].replace([np.inf, -np.inf], np.nan)
        y = sample[label]
        mean_up = x[sample[up] == 1].mean()
        mean_down = x[sample[up] == 0].mean()
        rows.append(
            {
                "factor": col,
                "corr_with_forward_return": signed_corr(x, y),
                "mean_when_up": mean_up,
                "mean_when_down": mean_down,
                "difference_up_down": mean_up - mean_down,
                "missing_rate": float(x.isna().mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_corr_rank"] = out["corr_with_forward_return"].abs().rank(ascending=False, method="dense")
    return out.sort_values(["abs_corr_rank", "factor"]).reset_index(drop=True)


def compute_factor_effectiveness_tail_targets(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        ret_col = f"target_ret_{h}d"
        big_up_col = f"target_big_up_{h}d"
        big_down_col = f"target_big_down_{h}d"
        clean_col = f"target_clean_direction_{h}d"
        end_col = f"target_end_date_{h}d"
        sample = df.loc[(df[end_col] < TEST_START) & df[ret_col].notna()].copy()
        for col in feature_cols:
            x = sample[col].replace([np.inf, -np.inf], np.nan)
            rows.append(
                {
                    "factor": col,
                    "horizon": h,
                    "corr_with_forward_return": signed_corr(x, sample[ret_col]),
                    "corr_with_big_up": signed_corr(x, sample[big_up_col]),
                    "corr_with_big_down": signed_corr(x, sample[big_down_col]),
                    "corr_with_clean_direction": signed_corr(x, sample[clean_col]),
                    "mean_when_big_up": x[sample[big_up_col] == 1].mean(),
                    "mean_when_big_down": x[sample[big_down_col] == 1].mean(),
                    "mean_when_clean_up": x[sample[clean_col] == 1].mean(),
                    "mean_when_clean_down": x[sample[clean_col] == 0].mean(),
                    "difference_big_up_down": x[sample[big_up_col] == 1].mean() - x[sample[big_down_col] == 1].mean(),
                    "missing_rate": float(x.isna().mean()),
                }
            )
    out = pd.DataFrame(rows)
    out["max_abs_tail_corr"] = out[["corr_with_big_up", "corr_with_big_down", "corr_with_clean_direction"]].abs().max(axis=1)
    return out.sort_values(["max_abs_tail_corr", "factor", "horizon"], ascending=[False, True, True]).reset_index(drop=True)
