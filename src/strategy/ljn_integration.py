from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    MIN_TRAIN_SAMPLES,
    RANDOM_STATE,
    RETRAIN_EVERY,
    TEST_END,
    TEST_START,
    TRADE_THRESHOLD,
    VALID_RETRAIN_EVERY,
)
from src.feature_selection import select_task_feature_columns
from src.models import build_model_specs, fit_binary_probability
from src.preprocessing import fit_preprocessor, transform_features
from src.strategy.backtest import backtest_position_frame, build_buy_hold_frame
from src.strategy.decision_rules import (
    FEATURE_CONTEXT_COLS,
    RiskBudgetParams,
    apply_no_trade_band_with_full_reset,
    ml_risk_budget_enhancement_position,
)
from src.strategy.metrics import compute_strategy_metrics
from src.walk_forward import walk_forward_predictions


LJN_OPPORTUNITY_COST_BUFFER = 0.0015
LJN_VALIDATION_FOLDS = [
    ("2023H2", pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-31")),
    ("2024H1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")),
    ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
]

LJN_BULL_STRATEGY = "ml_ljn_bull_full_participation_candidate"
LJN_SMOOTHING_STRATEGY = "ml_risk_budget_with_position_smoothing"
LJN_REDUCE_WORTH_STRATEGY = "ml_reduce_position_worth_enhancement"


@dataclass(frozen=True)
class LJNCandidate:
    config_id: str
    strategy_name: str
    model_name: str
    horizon: int
    params: dict[str, Any]


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["date", "trade_date", "target_start_date", "target_end_date", "chunk_start", "train_latest_target_end_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", format="mixed")
    return out


def add_ljn_reduce_position_targets(feature_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Add leakage-safe auxiliary labels used only as model targets.

    These columns are future outcomes, never features. Walk-forward training uses
    rows with target_end_date < chunk_start, and test rows only consume predicted
    probabilities.
    """
    out = feature_df.sort_values("date").copy()
    close = out["close"].astype(float)
    for horizon in horizons:
        ret = close.shift(-horizon) / close - 1
        future_path = pd.concat(
            [(close.shift(-step) / close - 1).rename(step) for step in range(1, horizon + 1)],
            axis=1,
        )
        future_drawdown = future_path.min(axis=1)
        valid = ret.notna() & future_drawdown.notna()
        out[f"ljn_future_drawdown_{horizon}d"] = future_drawdown
        out[f"target_ljn_avoid_loss_{horizon}d"] = np.where(
            valid,
            ((ret <= -TRADE_THRESHOLD) | (future_drawdown <= -TRADE_THRESHOLD)).astype(float),
            np.nan,
        )
        out[f"target_ljn_reduce_position_worth_{horizon}d"] = np.where(
            valid,
            (
                (future_drawdown <= -(TRADE_THRESHOLD + LJN_OPPORTUNITY_COST_BUFFER))
                & (ret <= LJN_OPPORTUNITY_COST_BUFFER)
            ).astype(float),
            np.nan,
        )
    return out


def apply_position_smoothing(
    position: pd.Series,
    max_position_change: float | None,
    initial_position: float = 1.0,
) -> pd.Series:
    if max_position_change is None or max_position_change >= 1.0:
        return position.astype(float)
    values: list[float] = []
    previous = float(initial_position)
    for raw in position.fillna(initial_position).astype(float):
        target = float(raw)
        if target > previous + max_position_change:
            current = previous + max_position_change
        elif target < previous - max_position_change:
            current = previous - max_position_change
        else:
            current = target
        current = min(1.0, max(0.0, current))
        values.append(current)
        previous = current
    return pd.Series(values, index=position.index, dtype=float)


def prepare_ljn_frame(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    preds = _parse_dates(predictions)
    pred = preds.loc[(preds["model_name"].astype(str) == model_name) & (preds["horizon"].astype(int) == int(horizon))].copy()
    if pred.empty:
        raise ValueError(f"No predictions for {model_name}, horizon={horizon}")
    context_base = feature_df.sort_values("date").copy()
    context_base["date"] = pd.to_datetime(context_base["date"], errors="coerce")
    if "crash_risk_score" in context_base.columns:
        context_base["crash_risk_score_roll60_70pct"] = (
            context_base["crash_risk_score"].shift(1).rolling(60, min_periods=20).quantile(0.70)
        )
    context_cols = [col for col in FEATURE_CONTEXT_COLS if col in context_base.columns]
    frame = pred.merge(context_base[context_cols], on="date", how="left")
    return frame.sort_values("date").dropna(subset=["fwd_ret_1d"]).reset_index(drop=True)


def _strong_up_trend(frame: pd.DataFrame) -> pd.Series:
    trend_score = frame.get("trend_regime_score", pd.Series(0.5, index=frame.index)).fillna(0.5)
    ret_20 = frame.get("ret_20", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ret_60 = frame.get("ret_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ma_ratio_20 = frame.get("ma_ratio_20", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ma_ratio_60 = frame.get("ma_ratio_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    return ((trend_score >= 0.62) & (ret_20 > 0.015) & (ret_60 > 0.03) & (ma_ratio_20 > 0) & (ma_ratio_60 > 0)).fillna(False)


def _market_risk(frame: pd.DataFrame, high_vol_multiplier: float = 1.15, drawdown_cut: float = -0.10) -> pd.Series:
    ma_ratio_60 = frame.get("ma_ratio_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    ret_20 = frame.get("ret_20", pd.Series(0.0, index=frame.index)).fillna(0.0)
    trend_score = frame.get("trend_regime_score", pd.Series(0.5, index=frame.index)).fillna(0.5)
    drawdown_60 = frame.get("drawdown_60", pd.Series(0.0, index=frame.index)).fillna(0.0)
    vol_ratio = frame.get("volatility_ratio_20_60", pd.Series(1.0, index=frame.index)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    trend_risk = (ma_ratio_60 < 0) | (ret_20 < 0) | (trend_score < 0.5)
    return (trend_risk | (drawdown_60 < drawdown_cut) | (vol_ratio > high_vol_multiplier)).fillna(False)


def ljn_bull_full_position(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    big_down = frame.get("pred_big_down_prob", pd.Series(0.5, index=frame.index)).fillna(0.5)
    up_prob = frame.get("pred_up_prob", pd.Series(0.5, index=frame.index)).fillna(0.5)
    pred_ret = frame.get("pred_ret", pd.Series(0.0, index=frame.index)).fillna(0.0)
    tail = frame.get("pred_tail_score", pd.Series(0.0, index=frame.index)).fillna(0.0)
    high_risk = _market_risk(frame, float(params["high_vol_multiplier"]), float(params["drawdown_cut"]))
    strong_up = _strong_up_trend(frame)
    downside = (
        (big_down >= float(params["big_down_cut"]))
        | (tail <= -abs(float(params["tail_cut"])))
        | (pred_ret < -0.004)
        | ((up_prob < 0.45) & high_risk)
    )
    severe = downside & high_risk & (frame.get("drawdown_60", pd.Series(0.0, index=frame.index)).fillna(0.0) < -0.14)
    raw = pd.Series(1.0, index=frame.index, dtype=float)
    raw = raw.mask(downside & high_risk & ~strong_up, float(params["defensive_position"]))
    raw = raw.mask(severe, float(params["severe_position"]))
    raw = raw.mask(strong_up & ~severe, 1.0)
    raw = raw.clip(lower=0.96, upper=1.0)
    banded = apply_no_trade_band_with_full_reset(raw, float(params["no_trade_band"]), initial_position=1.0)
    return apply_position_smoothing(banded, params.get("max_position_change"), initial_position=1.0)


def risk_budget_with_smoothing_position(
    frame: pd.DataFrame,
    risk_budget_params: RiskBudgetParams,
    max_position_change: float | None,
) -> pd.Series:
    base = ml_risk_budget_enhancement_position(frame.copy(), risk_budget_params)
    return apply_position_smoothing(base, max_position_change, initial_position=1.0)


def reduce_position_worth_position(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    reduce_prob = frame.get("pred_reduce_position_worth_prob", pd.Series(0.5, index=frame.index)).fillna(0.5)
    big_down = frame.get("pred_big_down_prob", pd.Series(0.5, index=frame.index)).fillna(0.5)
    pred_ret = frame.get("pred_ret", pd.Series(0.0, index=frame.index)).fillna(0.0)
    tail = frame.get("pred_tail_score", pd.Series(0.0, index=frame.index)).fillna(0.0)
    high_risk = _market_risk(frame, float(params["high_vol_multiplier"]), float(params["drawdown_cut"]))
    strong_up = _strong_up_trend(frame)
    reduce_signal = (
        (reduce_prob >= float(params["reduce_prob_cut"]))
        & ((big_down >= float(params["big_down_cut"])) | (tail < 0) | (pred_ret < LJN_OPPORTUNITY_COST_BUFFER) | high_risk)
        & ~strong_up
    )
    severe_signal = reduce_signal & high_risk & (reduce_prob >= float(params["severe_reduce_prob_cut"]))
    raw = pd.Series(1.0, index=frame.index, dtype=float)
    raw = raw.mask(reduce_signal, float(params["mild_position"]))
    raw = raw.mask(severe_signal, float(params["severe_position"]))
    raw = raw.mask(strong_up & ~severe_signal, 1.0)
    raw = raw.clip(lower=0.94, upper=1.0)
    banded = apply_no_trade_band_with_full_reset(raw, float(params["no_trade_band"]), initial_position=1.0)
    return apply_position_smoothing(banded, params.get("max_position_change"), initial_position=1.0)


def make_ljn_candidates(risk_budget_params: RiskBudgetParams) -> list[LJNCandidate]:
    candidates: list[LJNCandidate] = []
    for max_change in [None, 0.02, 0.05, 0.10]:
        label = "none" if max_change is None else str(max_change).replace(".", "p")
        candidates.append(
            LJNCandidate(
                config_id=f"smoothing_{label}",
                strategy_name=LJN_SMOOTHING_STRATEGY,
                model_name=risk_budget_params.model_name,
                horizon=int(risk_budget_params.horizon),
                params={"max_position_change": max_change},
            )
        )

    for idx, (big_down_cut, defensive_position, no_trade_band) in enumerate(
        product([0.62, 0.70], [0.990, 0.995], [0.03, 0.06]),
        start=1,
    ):
        candidates.append(
            LJNCandidate(
                config_id=f"bull_{idx:02d}",
                strategy_name=LJN_BULL_STRATEGY,
                model_name="ExtraTrees",
                horizon=10,
                params={
                    "big_down_cut": big_down_cut,
                    "tail_cut": 0.02,
                    "defensive_position": defensive_position,
                    "severe_position": 0.96,
                    "no_trade_band": no_trade_band,
                    "high_vol_multiplier": 1.15,
                    "drawdown_cut": -0.10,
                    "max_position_change": 0.30,
                },
            )
        )

    for idx, (reduce_prob_cut, mild_position, no_trade_band) in enumerate(
        product([0.55, 0.60, 0.65], [0.985, 0.990], [0.03, 0.06]),
        start=1,
    ):
        candidates.append(
            LJNCandidate(
                config_id=f"reduce_worth_{idx:02d}",
                strategy_name=LJN_REDUCE_WORTH_STRATEGY,
                model_name="ExtraTrees+ReduceWorthLogit",
                horizon=10,
                params={
                    "base_model_name": "ExtraTrees",
                    "reduce_prob_cut": reduce_prob_cut,
                    "severe_reduce_prob_cut": min(0.80, reduce_prob_cut + 0.15),
                    "big_down_cut": 0.58,
                    "mild_position": mild_position,
                    "severe_position": 0.96,
                    "no_trade_band": no_trade_band,
                    "high_vol_multiplier": 1.15,
                    "drawdown_cut": -0.10,
                    "max_position_change": 0.10,
                },
            )
        )
    return candidates


def generate_base_fold_predictions(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    fold_name: str,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    feature_top_k: int | None,
) -> pd.DataFrame:
    specs, _ = build_model_specs()
    wanted = {"LightGBM", "ExtraTrees"}
    selected_specs = [spec for spec in specs if spec.name in wanted]
    preds = walk_forward_predictions(
        feature_df,
        feature_cols,
        selected_specs,
        pd.Timestamp(fold_start),
        pd.Timestamp(fold_end),
        VALID_RETRAIN_EVERY,
        min_train_samples=MIN_TRAIN_SAMPLES,
        label=f"ljn_validation_{fold_name}",
        feature_top_k=feature_top_k,
    )
    return _parse_dates(preds)


def walk_forward_reduce_worth_predictions(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    horizon: int,
    label: str,
    feature_top_k: int | None = 50,
    retrain_every: int = RETRAIN_EVERY,
) -> pd.DataFrame:
    work = add_ljn_reduce_position_targets(feature_df, [horizon])
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    dates = (
        work.loc[(work["date"] >= pd.Timestamp(period_start)) & (work["date"] <= pd.Timestamp(period_end)), "date"]
        .dropna()
        .sort_values()
        .drop_duplicates()
        .tolist()
    )
    target_col = f"target_ljn_reduce_position_worth_{horizon}d"
    fallback_col = f"target_ret_{horizon}d"
    end_col = f"target_end_date_{horizon}d"
    rows: list[pd.DataFrame] = []
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    for chunk_idx in range(0, len(dates), retrain_every):
        chunk_dates = dates[chunk_idx : chunk_idx + retrain_every]
        if not chunk_dates:
            continue
        chunk_start = pd.Timestamp(chunk_dates[0])
        train_mask = (pd.to_datetime(work[end_col], errors="coerce") < chunk_start) & work[target_col].notna()
        test_mask = work["date"].isin(chunk_dates)
        train = work.loc[train_mask].copy()
        test = work.loc[test_mask].copy()
        if len(train) < MIN_TRAIN_SAMPLES or test.empty:
            continue
        selected_features = select_task_feature_columns(train, feature_cols, target_col, feature_top_k, fallback_col)
        valid_train = train.loc[train[target_col].notna()].copy()
        prep = fit_preprocessor(valid_train, selected_features)
        x_train = transform_features(valid_train, selected_features, prep)
        x_test = transform_features(test, selected_features, prep)
        y_train = valid_train[target_col].astype(int)
        if y_train.nunique() < 2:
            prob = np.full(len(test), float(y_train.mean()) if len(y_train) else 0.5)
        else:
            prob = fit_binary_probability(model, x_train, y_train, x_test)
        pred = test[["date", "trade_date", "close", "fwd_ret_1d", end_col, fallback_col, target_col]].copy()
        pred = pred.rename(
            columns={
                end_col: "target_end_date",
                fallback_col: "target_ret",
                target_col: "target_ljn_reduce_position_worth",
            }
        )
        pred["horizon"] = int(horizon)
        pred["model_name"] = "ExtraTrees+ReduceWorthLogit"
        pred["pred_reduce_position_worth_prob"] = prob
        pred["walk_forward_period"] = label
        pred["chunk_start"] = chunk_start
        pred["feature_top_k"] = "all" if feature_top_k is None else str(feature_top_k)
        rows.append(pred)
    if not rows:
        raise RuntimeError(f"No reduce-position-worth predictions generated for {label}.")
    return pd.concat(rows, ignore_index=True).sort_values(["date", "horizon"]).reset_index(drop=True)


def _candidate_frame(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    reduce_predictions: pd.DataFrame,
    candidate: LJNCandidate,
) -> pd.DataFrame:
    base_model = str(candidate.params.get("base_model_name", candidate.model_name))
    frame = prepare_ljn_frame(feature_df, predictions, base_model, candidate.horizon)
    if candidate.strategy_name == LJN_REDUCE_WORTH_STRATEGY:
        reduce_use = _parse_dates(reduce_predictions)[["date", "pred_reduce_position_worth_prob"]].copy()
        frame = frame.merge(reduce_use, on="date", how="left")
        frame["pred_reduce_position_worth_prob"] = frame["pred_reduce_position_worth_prob"].fillna(0.5)
    return frame


def _candidate_position(frame: pd.DataFrame, candidate: LJNCandidate, risk_budget_params: RiskBudgetParams) -> pd.Series:
    if candidate.strategy_name == LJN_SMOOTHING_STRATEGY:
        return risk_budget_with_smoothing_position(frame, risk_budget_params, candidate.params.get("max_position_change"))
    if candidate.strategy_name == LJN_BULL_STRATEGY:
        return ljn_bull_full_position(frame, candidate.params)
    if candidate.strategy_name == LJN_REDUCE_WORTH_STRATEGY:
        return reduce_position_worth_position(frame, candidate.params)
    raise ValueError(f"Unknown LJN candidate strategy: {candidate.strategy_name}")


def evaluate_candidate(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    reduce_predictions: pd.DataFrame,
    candidate: LJNCandidate,
    risk_budget_params: RiskBudgetParams,
    period_name: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> dict[str, Any]:
    frame = _candidate_frame(feature_df, predictions, reduce_predictions, candidate)
    position = _candidate_position(frame, candidate, risk_budget_params)
    signal = pd.Series(candidate.strategy_name, index=frame.index)
    strategy_daily = backtest_position_frame(frame, position, candidate.strategy_name, signal=signal, initial_position=1.0)
    metrics = compute_strategy_metrics(pd.concat([build_buy_hold_frame(frame), strategy_daily], ignore_index=True))
    row = metrics.loc[metrics["strategy"] == candidate.strategy_name].iloc[0].to_dict()
    return {
        "config_id": candidate.config_id,
        "strategy_name": candidate.strategy_name,
        "model_name": candidate.model_name,
        "horizon": int(candidate.horizon),
        "period_name": period_name,
        "fold_name": period_name,
        "fold_start": pd.Timestamp(period_start).date().isoformat(),
        "fold_end": pd.Timestamp(period_end).date().isoformat(),
        "params": candidate.params,
        "total_return": row.get("total_return"),
        "excess_return_vs_buy_hold": row.get("excess_return_vs_buy_hold"),
        "max_drawdown": row.get("max_drawdown"),
        "sharpe": row.get("sharpe"),
        "calmar": row.get("calmar"),
        "average_position": row.get("average_position"),
        "total_turnover": row.get("total_turnover"),
        "transaction_cost": row.get("total_transaction_cost"),
        "missed_upside": row.get("missed_upside"),
        "avoided_downside": row.get("avoided_downside"),
        "benchmark_clone": row.get("benchmark_clone"),
    }


def aggregate_ljn_results(
    validation_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    current_main_total_return: float,
) -> pd.DataFrame:
    grouped = validation_rows.groupby(["config_id", "strategy_name", "model_name", "horizon"], as_index=False)
    agg = grouped.agg(
        mean_validation_excess=("excess_return_vs_buy_hold", "mean"),
        min_validation_excess=("excess_return_vs_buy_hold", "min"),
        std_validation_excess=("excess_return_vs_buy_hold", "std"),
        mean_validation_calmar=("calmar", "mean"),
        min_validation_calmar=("calmar", "min"),
        mean_turnover=("total_turnover", "mean"),
        mean_validation_position=("average_position", "mean"),
    )
    agg["std_validation_excess"] = agg["std_validation_excess"].fillna(0.0)
    agg["robust_score"] = (
        agg["mean_validation_excess"].fillna(0.0)
        + 0.5 * agg["min_validation_excess"].fillna(0.0)
        + 0.2 * agg["mean_validation_calmar"].fillna(0.0)
        - 0.1 * agg["std_validation_excess"].fillna(0.0)
        - 0.05 * agg["mean_turnover"].fillna(0.0)
    )
    test_use = test_rows.rename(
        columns={
            "total_return": "test_total_return",
            "excess_return_vs_buy_hold": "test_excess_return_vs_buy_hold",
            "max_drawdown": "test_max_drawdown",
            "sharpe": "test_sharpe",
            "calmar": "test_calmar",
            "average_position": "test_average_position",
            "total_turnover": "test_turnover",
        }
    )[
        [
            "config_id",
            "test_total_return",
            "test_excess_return_vs_buy_hold",
            "test_max_drawdown",
            "test_sharpe",
            "test_calmar",
            "test_average_position",
            "test_turnover",
        ]
    ]
    out = agg.merge(test_use, on="config_id", how="left")
    selected_config = out.sort_values("robust_score", ascending=False).iloc[0]["config_id"] if not out.empty else None
    out["selected_by_multifold_validation"] = out["config_id"] == selected_config
    out["beats_current_main_strategy"] = out["test_total_return"] > float(current_main_total_return)
    out["beats_buy_hold"] = out["test_excess_return_vs_buy_hold"] > 0
    out["final_recommendation"] = np.where(
        out["selected_by_multifold_validation"] & out["beats_current_main_strategy"],
        "validation-selected candidate also beats current main; review before upgrade",
        np.where(
            out["selected_by_multifold_validation"],
            "validation-selected supplement; do not upgrade main strategy",
            "not selected by multifold validation",
        ),
    )
    return out.sort_values(["selected_by_multifold_validation", "robust_score"], ascending=[False, False]).reset_index(drop=True)


def run_ljn_integration_experiment(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    risk_budget_params: RiskBudgetParams,
    feature_top_k: int | None,
    current_main_total_return: float,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = cache_dir or Path("outputs/current/misc")
    cache_dir.mkdir(parents=True, exist_ok=True)
    validation_predictions = _parse_dates(validation_predictions)
    test_predictions = _parse_dates(test_predictions)
    feature_df = feature_df.copy()
    feature_df["date"] = pd.to_datetime(feature_df["date"], errors="coerce")

    base_by_fold: dict[str, pd.DataFrame] = {}
    reduce_by_fold: dict[str, pd.DataFrame] = {}
    for fold_name, fold_start, fold_end in LJN_VALIDATION_FOLDS:
        if fold_name == "2024H2":
            base_by_fold[fold_name] = validation_predictions
        else:
            cache_path = cache_dir / f"ljn_base_predictions_{fold_name}.csv"
            if cache_path.exists():
                base_by_fold[fold_name] = _parse_dates(pd.read_csv(cache_path))
            else:
                preds = generate_base_fold_predictions(feature_df, feature_cols, fold_name, fold_start, fold_end, feature_top_k)
                preds.to_csv(cache_path, index=False)
                base_by_fold[fold_name] = preds

        reduce_cache = cache_dir / f"ljn_reduce_worth_predictions_{fold_name}.csv"
        if reduce_cache.exists():
            reduce_by_fold[fold_name] = _parse_dates(pd.read_csv(reduce_cache))
        else:
            reduce_preds = walk_forward_reduce_worth_predictions(
                feature_df,
                feature_cols,
                fold_start,
                fold_end,
                horizon=10,
                label=f"ljn_reduce_{fold_name}",
                feature_top_k=50,
                retrain_every=VALID_RETRAIN_EVERY,
            )
            reduce_preds.to_csv(reduce_cache, index=False)
            reduce_by_fold[fold_name] = reduce_preds

    test_reduce_cache = cache_dir / "ljn_reduce_worth_predictions_test.csv"
    if test_reduce_cache.exists():
        reduce_test = _parse_dates(pd.read_csv(test_reduce_cache))
    else:
        reduce_test = walk_forward_reduce_worth_predictions(
            feature_df,
            feature_cols,
            TEST_START,
            TEST_END,
            horizon=10,
            label="ljn_reduce_test",
            feature_top_k=50,
            retrain_every=RETRAIN_EVERY,
        )
        reduce_test.to_csv(test_reduce_cache, index=False)

    candidates = make_ljn_candidates(risk_budget_params)
    validation_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for candidate in candidates:
        for fold_name, fold_start, fold_end in LJN_VALIDATION_FOLDS:
            validation_records.append(
                evaluate_candidate(
                    feature_df,
                    base_by_fold[fold_name],
                    reduce_by_fold[fold_name],
                    candidate,
                    risk_budget_params,
                    fold_name,
                    fold_start,
                    fold_end,
                )
            )
        test_records.append(
            evaluate_candidate(
                feature_df,
                test_predictions,
                reduce_test,
                candidate,
                risk_budget_params,
                "test",
                TEST_START,
                TEST_END,
            )
        )

    validation_df = pd.DataFrame(validation_records)
    test_df = pd.DataFrame(test_records)
    aggregate = aggregate_ljn_results(validation_df, test_df, current_main_total_return)
    validation_df = validation_df.merge(
        aggregate[["config_id", "robust_score", "selected_by_multifold_validation"]],
        on="config_id",
        how="left",
    )
    return validation_df, aggregate
