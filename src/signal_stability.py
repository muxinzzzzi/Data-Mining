from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils import rolling_percentile


DEFAULT_SIGNAL_COLS = [
    "pred_trade_prob",
    "pred_big_up_prob",
    "pred_tail_score",
    "pred_ret",
    "pred_up_prob",
]

UPWARD_SIGNAL_COLS = [
    "pred_trade_prob",
    "pred_big_up_prob",
    "pred_tail_score",
    "pred_ret",
    "pred_up_prob",
]

SELECTABLE_UPSIDE_SIGNALS = [
    "pred_trade_prob",
    "pred_big_up_prob",
    "pred_up_prob",
    "pred_tail_score",
]

STABILITY_AWARE_UPSIDE_SIGNALS = [
    "pred_trade_prob",
    "pred_big_up_prob",
    "pred_up_prob",
]


def _period_label(value: Any) -> str:
    text = str(value)
    if text.startswith("validation"):
        return "validation"
    if text.startswith("test"):
        return "test"
    return text if text and text != "nan" else "unknown"


def _assign_period(pred_df: pd.DataFrame, periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None) -> pd.DataFrame:
    out = pred_df.copy()
    if "period" in out.columns:
        out["period"] = out["period"].map(_period_label)
        return out
    if "diagnostic_period" in out.columns:
        out["period"] = out["diagnostic_period"].map(_period_label)
        return out
    if "walk_forward_period" in out.columns:
        out["period"] = out["walk_forward_period"].map(_period_label)
        return out
    if periods:
        out["date"] = pd.to_datetime(out["date"])
        out["period"] = "unknown"
        for name, (start, end) in periods.items():
            mask = (out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))
            out.loc[mask, "period"] = name
        return out
    out["period"] = "combined"
    return out


def _make_quantile_groups(group: pd.DataFrame, signal_name: str, n_quantiles: int) -> pd.Series:
    signal = group[signal_name].replace([np.inf, -np.inf], np.nan)
    ret = group["fwd_ret_1d"].replace([np.inf, -np.inf], np.nan)
    valid = signal.notna() & ret.notna()
    labels = pd.Series(np.nan, index=group.index, dtype=float)
    if int(valid.sum()) < n_quantiles:
        return labels
    ranked = signal.loc[valid].rank(method="first")
    labels.loc[valid] = pd.qcut(ranked, n_quantiles, labels=False, duplicates="drop").astype(float) + 1
    return labels


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    return float(df[col].replace([np.inf, -np.inf], np.nan).mean())


def compute_signal_quantile_stability(
    pred_df: pd.DataFrame,
    signal_cols: list[str] | None = None,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    n_quantiles: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure whether higher prediction quantiles map to higher next-day returns.

    The function consumes prediction outputs only. It does not choose models from
    test data; test rows are diagnostics when present.
    """
    signal_cols = signal_cols or DEFAULT_SIGNAL_COLS
    required = {"date", "model_name", "horizon", "fwd_ret_1d"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction data missing columns for stability diagnostics: {sorted(missing)}")

    available_signals = [col for col in signal_cols if col in pred_df.columns]
    if not available_signals:
        raise ValueError("No requested signal columns are available in prediction data.")

    work = _assign_period(pred_df, periods)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    bucket_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for (period, model_name, horizon), grp in work.groupby(["period", "model_name", "horizon"], sort=True):
        if grp.empty:
            continue
        for signal_name in available_signals:
            bucketed = grp.copy()
            bucketed["quantile_group"] = _make_quantile_groups(bucketed, signal_name, n_quantiles)
            bucketed = bucketed.dropna(subset=["quantile_group", "fwd_ret_1d", signal_name])
            if bucketed.empty:
                continue

            group_means: list[float] = []
            group_ids: list[int] = []
            for quantile_group, qgrp in bucketed.groupby("quantile_group", sort=True):
                quantile = int(quantile_group)
                returns = qgrp["fwd_ret_1d"].astype(float)
                mean_return = float(returns.mean())
                group_ids.append(quantile)
                group_means.append(mean_return)
                bucket_rows.append(
                    {
                        "period": period,
                        "model_name": model_name,
                        "horizon": int(horizon),
                        "signal_name": signal_name,
                        "quantile_group": f"Q{quantile}",
                        "quantile_index": quantile,
                        "count": int(len(qgrp)),
                        "mean_forward_return": mean_return,
                        "median_forward_return": float(returns.median()),
                        "win_rate": float((returns > 0).mean()),
                        "mean_target_trade": _safe_mean(qgrp, "target_trade"),
                        "mean_target_big_up": _safe_mean(qgrp, "target_big_up"),
                        "mean_target_big_down": _safe_mean(qgrp, "target_big_down"),
                        "avg_signal_value": float(qgrp[signal_name].astype(float).mean()),
                    }
                )

            by_quantile = pd.DataFrame(
                {
                    "quantile_index": group_ids,
                    "mean_forward_return": group_means,
                }
            ).sort_values("quantile_index")
            bottom = by_quantile.loc[by_quantile["quantile_index"] == 1]
            top = by_quantile.loc[by_quantile["quantile_index"] == n_quantiles]
            if bottom.empty or top.empty:
                continue
            top_rows = bucketed.loc[bucketed["quantile_group"] == n_quantiles]
            bottom_rows = bucketed.loc[bucketed["quantile_group"] == 1]
            monotonicity = (
                float(by_quantile["quantile_index"].corr(by_quantile["mean_forward_return"], method="spearman"))
                if len(by_quantile) >= 2 and by_quantile["mean_forward_return"].nunique(dropna=True) > 1
                else np.nan
            )
            top_mean = float(top.iloc[0]["mean_forward_return"])
            bottom_mean = float(bottom.iloc[0]["mean_forward_return"])
            summary_rows.append(
                {
                    "period": period,
                    "model_name": model_name,
                    "horizon": int(horizon),
                    "signal_name": signal_name,
                    "top_group_mean_return": top_mean,
                    "bottom_group_mean_return": bottom_mean,
                    "top_bottom_spread": top_mean - bottom_mean,
                    "top_group_win_rate": float((top_rows["fwd_ret_1d"] > 0).mean()),
                    "bottom_group_win_rate": float((bottom_rows["fwd_ret_1d"] > 0).mean()),
                    "monotonicity_score": monotonicity,
                    "top_group_trade_rate": _safe_mean(top_rows, "target_trade"),
                    "bottom_group_trade_rate": _safe_mean(bottom_rows, "target_trade"),
                    "top_group_big_up_rate": _safe_mean(top_rows, "target_big_up"),
                    "bottom_group_big_up_rate": _safe_mean(bottom_rows, "target_big_up"),
                    "top_group_big_down_rate": _safe_mean(top_rows, "target_big_down"),
                    "bottom_group_big_down_rate": _safe_mean(bottom_rows, "target_big_down"),
                    "bottom_group_negative_mean": bool(bottom_mean < 0),
                    "top_group_count": int(len(top_rows)),
                    "bottom_group_count": int(len(bottom_rows)),
                }
            )

    bucket_df = pd.DataFrame(bucket_rows)
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return bucket_df, summary_df

    spreads = summary_df.pivot_table(
        index=["model_name", "horizon", "signal_name"],
        columns="period",
        values="top_bottom_spread",
        aggfunc="first",
    )
    if "validation" in spreads.columns and "test" in spreads.columns:
        spreads["validation_to_test_spread_decay"] = spreads["test"] - spreads["validation"]
        summary_df = summary_df.merge(
            spreads["validation_to_test_spread_decay"].reset_index(),
            on=["model_name", "horizon", "signal_name"],
            how="left",
        )
    else:
        summary_df["validation_to_test_spread_decay"] = np.nan

    return (
        bucket_df.sort_values(["period", "model_name", "horizon", "signal_name", "quantile_index"]).reset_index(drop=True),
        summary_df.sort_values(["period", "model_name", "horizon", "signal_name"]).reset_index(drop=True),
    )


def select_tradeable_upside_model(
    model_metrics: pd.DataFrame,
    signal_stability_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Select an upside signal model using validation-period metrics only."""
    validation_stability = signal_stability_summary.loc[
        (signal_stability_summary["period"] == "validation")
        & (signal_stability_summary["signal_name"].isin(SELECTABLE_UPSIDE_SIGNALS))
    ].copy()
    if validation_stability.empty:
        raise ValueError("Validation signal stability summary is empty.")

    metrics = model_metrics.copy()
    merged = validation_stability.merge(metrics, on=["model_name", "horizon"], how="left", suffixes=("", "_metric"))
    for col in ["trade_AUC", "big_up_AUC", "return_correlation"]:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged.loc[merged["top_bottom_spread"].fillna(-np.inf) > 0].copy()
    if merged.empty:
        raise ValueError("No validation signal has positive top-bottom spread; cannot select upside signal model.")

    merged["stability_proxy"] = merged["monotonicity_score"].fillna(0.0)
    merged["selection_score"] = (
        0.25 * merged["trade_AUC"].fillna(0.5)
        + 0.25 * merged["big_up_AUC"].fillna(0.5)
        + 0.20 * merged["top_bottom_spread"].clip(lower=0).fillna(0.0)
        + 0.15 * merged["stability_proxy"].clip(lower=0).fillna(0.0)
        + 0.10 * merged["top_group_win_rate"].fillna(0.5)
        + 0.05 * merged["return_correlation"].clip(lower=0).fillna(0.0)
    )
    priority = {
        "pred_trade_prob": 0,
        "pred_big_up_prob": 0,
        "pred_up_prob": 1,
        "pred_tail_score": 2,
    }
    merged["selection_tier"] = merged["signal_name"].map(priority).fillna(3).astype(int)
    merged["selection_tier_note"] = np.where(
        merged["selection_tier"] == 0,
        "preferred_tradeable_upside_signal",
        np.where(
            merged["selection_tier"] == 1,
            "fallback_up_probability_signal",
            "tail_score_allowed_only_after_positive_validation_spread",
        ),
    )
    ranked_all = merged.sort_values(
        ["selection_tier", "selection_score", "top_bottom_spread", "monotonicity_score"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    selected = ranked_all.iloc[0].to_dict()
    export_candidates = ranked_all.drop(columns=["validation_to_test_spread_decay"], errors="ignore")
    top_candidates = export_candidates.head(20).replace([np.inf, -np.inf], np.nan).to_dict(orient="records")
    return {
        "selected": {
            "model_name": selected["model_name"],
            "horizon": int(selected["horizon"]),
            "signal_name": selected["signal_name"],
            "score": float(selected["selection_score"]),
            "top_bottom_spread": float(selected["top_bottom_spread"]),
            "top_group_win_rate": float(selected["top_group_win_rate"]),
            "monotonicity_score": float(selected["monotonicity_score"]),
            "trade_AUC": float(selected["trade_AUC"]) if pd.notna(selected["trade_AUC"]) else None,
            "big_up_AUC": float(selected["big_up_AUC"]) if pd.notna(selected["big_up_AUC"]) else None,
            "selection_tier_note": selected["selection_tier_note"],
        },
        "top_candidates": top_candidates,
        "selection_rule": (
            "Validation-period only. Candidate signals must have positive validation top-bottom spread. "
            "Preferred tier is pred_trade_prob/pred_big_up_prob, fallback is pred_up_prob, and pred_tail_score "
            "is allowed only after positive validation spread."
        ),
        "score_formula": (
            "0.25*trade_AUC + 0.25*big_up_AUC + 0.20*max(0, validation_top_bottom_spread) + "
            "0.15*max(0, monotonicity_score) + 0.10*top_group_win_rate + "
            "0.05*max(0, return_correlation)"
        ),
        "note": "Model and signal selection is based only on validation-period signal structure and validation metrics.",
    }


def _penalty_breakdown(row: pd.Series) -> dict[str, float]:
    model_name = str(row.get("model_name", ""))
    horizon = int(row.get("horizon", 0))
    top_group_count = float(row.get("top_group_count", 0.0) or 0.0)
    monotonicity = float(row.get("monotonicity_score", 0.0) or 0.0)
    bottom_negative = bool(row.get("bottom_group_negative_mean", False))
    breakdown = {
        "deep_model_penalty": 0.04 if "Deep" in model_name else 0.0,
        "horizon_10_penalty": 0.015 if horizon == 10 else 0.0,
        "horizon_20_penalty": 0.035 if horizon == 20 else 0.0,
        "positive_bottom_group_penalty": 0.0 if bottom_negative else 0.02,
        "low_monotonicity_penalty": 0.03 if monotonicity < 0.5 else 0.0,
        "small_top_group_penalty": 0.01 if top_group_count < 30 else 0.0,
    }
    breakdown["total_penalty"] = float(sum(breakdown.values()))
    return breakdown


def select_stability_aware_upside_signals(
    validation_signal_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
    max_components: int = 3,
) -> dict[str, Any]:
    """Select multiple validation-only upside signals with stability penalties.

    Test-period diagnostics are intentionally not accepted by this function.
    """
    stability = validation_signal_summary.loc[
        (validation_signal_summary["period"] == "validation")
        & (validation_signal_summary["signal_name"].isin(STABILITY_AWARE_UPSIDE_SIGNALS))
    ].copy()
    if stability.empty:
        raise ValueError("No validation stability rows are available for stability-aware signal selection.")

    metrics = model_metrics.copy()
    merged = stability.merge(metrics, on=["model_name", "horizon"], how="left", suffixes=("", "_metric"))
    for col in ["AUC", "trade_AUC", "big_up_AUC", "return_correlation", "n_eval"]:
        if col not in merged.columns:
            merged[col] = np.nan

    filtered = merged.loc[
        (merged["top_bottom_spread"].fillna(-np.inf) > 0)
        & (merged["top_group_mean_return"] > merged["bottom_group_mean_return"])
        & (merged["top_group_trade_rate"].fillna(-np.inf) >= merged["bottom_group_trade_rate"].fillna(np.inf))
        & (merged["top_group_count"].fillna(0) >= 20)
        & (merged["n_eval"].fillna(0) >= 100)
        & (
            (merged["AUC"].fillna(0.5) > 0.55)
            | (merged["trade_AUC"].fillna(0.5) > 0.55)
            | (merged["big_up_AUC"].fillna(0.5) > 0.55)
        )
    ].copy()
    if filtered.empty:
        raise ValueError("No stability-aware upside signal candidates pass validation-only filters.")

    penalty_rows = filtered.apply(_penalty_breakdown, axis=1)
    filtered["penalty_breakdown"] = list(penalty_rows)
    filtered["complexity_penalty"] = [item["total_penalty"] for item in penalty_rows]
    filtered["raw_stable_score"] = (
        0.22 * filtered["trade_AUC"].fillna(0.5)
        + 0.22 * filtered["big_up_AUC"].fillna(0.5)
        + 0.18 * filtered["AUC"].fillna(0.5)
        + 0.14 * filtered["top_bottom_spread"].clip(lower=0).fillna(0.0)
        + 0.10 * filtered["monotonicity_score"].fillna(0.0)
        + 0.08 * filtered["top_group_win_rate"].fillna(0.5)
        + 0.06 * filtered["return_correlation"].clip(lower=0).fillna(0.0)
    )
    filtered["stable_score"] = filtered["raw_stable_score"] - filtered["complexity_penalty"]
    priority = {"pred_trade_prob": 0, "pred_big_up_prob": 0, "pred_up_prob": 1}
    filtered["signal_priority"] = filtered["signal_name"].map(priority).fillna(2).astype(int)
    ranked = filtered.sort_values(
        ["stable_score", "signal_priority", "top_bottom_spread", "monotonicity_score"],
        ascending=[False, True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    selected_rows: list[pd.Series] = []
    used_keys: set[tuple[str, int, str]] = set()
    used_signals: set[str] = set()
    for _, row in ranked.iterrows():
        key = (str(row["model_name"]), int(row["horizon"]), str(row["signal_name"]))
        if key in used_keys:
            continue
        if row["signal_name"] in used_signals and len(used_signals) < len(STABILITY_AWARE_UPSIDE_SIGNALS):
            continue
        selected_rows.append(row)
        used_keys.add(key)
        used_signals.add(str(row["signal_name"]))
        if len(selected_rows) >= max_components:
            break
    if len(selected_rows) < max_components:
        for _, row in ranked.iterrows():
            key = (str(row["model_name"]), int(row["horizon"]), str(row["signal_name"]))
            if key in used_keys:
                continue
            selected_rows.append(row)
            used_keys.add(key)
            if len(selected_rows) >= max_components:
                break

    def export_component(row: pd.Series) -> dict[str, Any]:
        return {
            "model_name": str(row["model_name"]),
            "horizon": int(row["horizon"]),
            "signal_name": str(row["signal_name"]),
            "stable_score": float(row["stable_score"]),
            "raw_stable_score": float(row["raw_stable_score"]),
            "complexity_penalty": float(row["complexity_penalty"]),
            "penalty_breakdown": row["penalty_breakdown"],
            "raw_metrics": {
                "AUC": float(row["AUC"]) if pd.notna(row["AUC"]) else None,
                "trade_AUC": float(row["trade_AUC"]) if pd.notna(row["trade_AUC"]) else None,
                "big_up_AUC": float(row["big_up_AUC"]) if pd.notna(row["big_up_AUC"]) else None,
                "return_correlation": float(row["return_correlation"]) if pd.notna(row["return_correlation"]) else None,
                "n_eval": int(row["n_eval"]) if pd.notna(row["n_eval"]) else None,
                "top_bottom_spread": float(row["top_bottom_spread"]),
                "top_group_mean_return": float(row["top_group_mean_return"]),
                "bottom_group_mean_return": float(row["bottom_group_mean_return"]),
                "top_group_win_rate": float(row["top_group_win_rate"]),
                "bottom_group_win_rate": float(row["bottom_group_win_rate"]),
                "monotonicity_score": float(row["monotonicity_score"]) if pd.notna(row["monotonicity_score"]) else None,
                "top_group_trade_rate": float(row["top_group_trade_rate"]) if pd.notna(row["top_group_trade_rate"]) else None,
                "bottom_group_trade_rate": float(row["bottom_group_trade_rate"]) if pd.notna(row["bottom_group_trade_rate"]) else None,
                "top_group_big_up_rate": float(row["top_group_big_up_rate"]) if pd.notna(row["top_group_big_up_rate"]) else None,
                "bottom_group_big_up_rate": float(row["bottom_group_big_up_rate"]) if pd.notna(row["bottom_group_big_up_rate"]) else None,
                "bottom_group_negative_mean": bool(row["bottom_group_negative_mean"]),
                "top_group_count": int(row["top_group_count"]),
            },
        }

    selected_components = [export_component(row) for row in selected_rows]
    candidate_cols = [
        "model_name",
        "horizon",
        "signal_name",
        "stable_score",
        "raw_stable_score",
        "complexity_penalty",
        "penalty_breakdown",
        "AUC",
        "trade_AUC",
        "big_up_AUC",
        "return_correlation",
        "n_eval",
        "top_bottom_spread",
        "top_group_mean_return",
        "bottom_group_mean_return",
        "top_group_win_rate",
        "bottom_group_win_rate",
        "monotonicity_score",
        "top_group_trade_rate",
        "bottom_group_trade_rate",
        "top_group_count",
        "bottom_group_negative_mean",
    ]
    top_candidates = ranked.loc[:, [col for col in candidate_cols if col in ranked.columns]].head(20)
    return {
        "selected_components": selected_components,
        "top_candidates": top_candidates.replace([np.inf, -np.inf], np.nan).to_dict(orient="records"),
        "selection_rule": (
            "Validation-period only. Signals are limited to pred_trade_prob, pred_big_up_prob, and pred_up_prob. "
            "Candidates must have positive validation top-bottom spread, top trade rate at least bottom trade rate, "
            "top-group count >=20, n_eval >=100, and at least one of AUC/trade_AUC/big_up_AUC above 0.55. "
            "Stable score uses validation metrics with penalties for Deep models, longer horizons, positive bottom groups, "
            "low monotonicity, and small top groups."
        ),
        "score_formula": (
            "0.22*trade_AUC + 0.22*big_up_AUC + 0.18*AUC + 0.14*max(0, top_bottom_spread) + "
            "0.10*monotonicity_score + 0.08*top_group_win_rate + 0.06*max(0, return_correlation) - complexity_penalty"
        ),
        "warning": "Selected only from validation period; test is reserved for final evaluation.",
    }


def _component_col_name(index: int, component: dict[str, Any]) -> str:
    signal = str(component["signal_name"]).replace("pred_", "").replace("_prob", "")
    model = str(component["model_name"]).replace(" ", "_")
    horizon = int(component["horizon"])
    return f"component_{index}_{model}_{horizon}d_{signal}_rank"


def build_stability_aware_upside_score(
    pred_all: pd.DataFrame,
    selected_components: list[dict[str, Any]],
) -> pd.DataFrame:
    if not selected_components:
        raise ValueError("No selected components were provided.")
    frames: list[pd.DataFrame] = []
    weights: list[float] = []
    for idx, component in enumerate(selected_components, start=1):
        model_name = component["model_name"]
        horizon = int(component["horizon"])
        signal_name = component["signal_name"]
        if signal_name not in pred_all.columns:
            raise ValueError(f"Selected signal is missing from predictions: {signal_name}")
        comp = pred_all.loc[
            (pred_all["model_name"] == model_name) & (pred_all["horizon"] == horizon),
            ["date", signal_name],
        ].copy()
        if comp.empty:
            raise ValueError(f"No predictions for selected component: {model_name}, {horizon}D, {signal_name}")
        comp["date"] = pd.to_datetime(comp["date"], errors="coerce")
        comp = comp.sort_values("date").drop_duplicates("date")
        rank_col = _component_col_name(idx, component)
        comp[rank_col] = rolling_percentile(comp[signal_name].astype(float), 60, 20).fillna(0.5)
        comp = comp.rename(columns={signal_name: f"{rank_col}_raw_signal"})
        frames.append(comp[["date", rank_col, f"{rank_col}_raw_signal"]])
        try:
            weights.append(max(0.0, float(component.get("stable_score", 0.0))))
        except (TypeError, ValueError):
            weights.append(0.0)

    score = frames[0]
    for frame in frames[1:]:
        score = score.merge(frame, on="date", how="outer")
    score = score.sort_values("date").reset_index(drop=True)
    rank_cols = [_component_col_name(idx, component) for idx, component in enumerate(selected_components, start=1)]
    for col in rank_cols:
        score[col] = score[col].fillna(0.5)
    if sum(weights) > 0:
        normalized = np.array(weights, dtype=float) / sum(weights)
    else:
        normalized = np.full(len(rank_cols), 1.0 / len(rank_cols), dtype=float)
    score["ensemble_upside_score"] = sum(score[col] * weight for col, weight in zip(rank_cols, normalized))
    score["ensemble_component_weight_sum"] = float(normalized.sum())
    for col, weight in zip(rank_cols, normalized):
        score[f"{col}_weight"] = float(weight)
    return score


def markdown_signal_stability_report(
    summary: pd.DataFrame,
    selection: dict[str, Any],
) -> str:
    def _fmt(value: Any, digits: int = 6) -> str:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "NA"
        if not np.isfinite(val):
            return "NA"
        return f"{val:.{digits}f}"

    def _table(df: pd.DataFrame, cols: list[str], max_rows: int = 12) -> str:
        if df.empty:
            return "No rows."
        use = df.loc[:, [col for col in cols if col in df.columns]].head(max_rows)
        lines = [
            "| " + " | ".join(use.columns) + " |",
            "| " + " | ".join(["---"] * len(use.columns)) + " |",
        ]
        for row in use.itertuples(index=False):
            values = []
            for value in row:
                values.append(_fmt(value) if isinstance(value, (float, np.floating)) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    selected = selection["selected"]
    selected_rows = summary.loc[
        (summary["model_name"] == selected["model_name"])
        & (summary["horizon"] == int(selected["horizon"]))
        & (summary["signal_name"] == selected["signal_name"])
    ].copy()
    validation = selected_rows.loc[selected_rows["period"] == "validation"]
    test = selected_rows.loc[selected_rows["period"] == "test"]

    validation_rank = summary.loc[summary["period"] == "validation"].sort_values(
        ["top_bottom_spread", "monotonicity_score"],
        ascending=[False, False],
        na_position="last",
    )
    test_rank = summary.loc[summary["period"] == "test"].sort_values(
        ["top_bottom_spread", "monotonicity_score"],
        ascending=[False, False],
        na_position="last",
    )
    val_top = validation_rank.iloc[0].to_dict() if not validation_rank.empty else {}
    test_top = test_rank.iloc[0].to_dict() if not test_rank.empty else {}

    upward = summary.loc[(summary["period"].isin(["validation", "test"])) & (summary["signal_name"].isin(UPWARD_SIGNAL_COLS))]
    low_negative_rate = float(upward["bottom_group_negative_mean"].mean()) if not upward.empty else np.nan
    selected_low_negative = bool(
        not validation.empty
        and not test.empty
        and bool(validation.iloc[0].get("bottom_group_negative_mean", False))
        and bool(test.iloc[0].get("bottom_group_negative_mean", False))
    )
    low_negative_stable = bool(np.isfinite(low_negative_rate) and low_negative_rate >= 0.60 and selected_low_negative)
    low_conclusion = (
        "Low-score groups show negative future returns often enough to support defensive selling."
        if low_negative_stable
        else "Low-score groups do not reliably correspond to future declines, so the strategy should not depend on low-score forced selling."
    )

    return f"""# Signal Quantile Stability Summary

## Selected Tradeable-Up Signal
- Selected signal: {selected["signal_name"]}
- Selected model/horizon: {selected["model_name"]}, {selected["horizon"]}D
- Validation top-bottom spread: {_fmt(validation.iloc[0]["top_bottom_spread"] if not validation.empty else np.nan)}
- Validation top group win rate: {_fmt(validation.iloc[0]["top_group_win_rate"] if not validation.empty else np.nan)}
- Test top-bottom spread: {_fmt(test.iloc[0]["top_bottom_spread"] if not test.empty else np.nan)}
- Test top group win rate: {_fmt(test.iloc[0]["top_group_win_rate"] if not test.empty else np.nan)}

## Most Stable Top-Bottom Spread
- Best validation spread: {val_top.get("signal_name", "NA")} / {val_top.get("model_name", "NA")} / {val_top.get("horizon", "NA")}D, spread {_fmt(val_top.get("top_bottom_spread"))}
- Best test spread: {test_top.get("signal_name", "NA")} / {test_top.get("model_name", "NA")} / {test_top.get("horizon", "NA")}D, spread {_fmt(test_top.get("top_bottom_spread"))}
- Share of upward-signal low groups with negative mean next-day return: {_fmt(low_negative_rate)}

## Validation Leaders
{_table(validation_rank, [
    "signal_name",
    "model_name",
    "horizon",
    "top_bottom_spread",
    "top_group_win_rate",
    "top_group_trade_rate",
    "top_group_big_up_rate",
    "monotonicity_score",
    "validation_to_test_spread_decay",
])}

## Test Diagnostics
{_table(test_rank, [
    "signal_name",
    "model_name",
    "horizon",
    "top_bottom_spread",
    "top_group_win_rate",
    "top_group_trade_rate",
    "top_group_big_up_rate",
    "monotonicity_score",
])}

## Low-Score Interpretation
{low_conclusion}

Test-period rows above are used only for post-selection diagnosis. The upside signal model is selected from validation-period statistics only.
"""
