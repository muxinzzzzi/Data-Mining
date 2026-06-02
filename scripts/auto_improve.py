from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import traceback
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    DIAGNOSTICS_DIR,
    HORIZONS,
    METRICS_DIR,
    MIN_TRAIN_SAMPLES,
    OUTPUT_DIR,
    PREDICTION_DIR,
    RETRAIN_EVERY,
    TEST_END,
    TEST_START,
    VALIDATION_FOLDS,
)
from src.data_loader import load_data
from src.features import build_daily_features, build_intraday_features, get_feature_columns
from src.metrics import compute_model_metrics
from src.models import ModelSpec, build_model_specs
from src.strategy.backtest import backtest_position_frame, build_buy_hold_frame
from src.strategy.decision_rules import StrategyFamily, StrategyParams, build_strategy_positions, prepare_strategy_frame
from src.strategy.metrics import compute_strategy_metrics, robust_score_details_from_fold_metrics
from src.utils import write_json_ready
from src.walk_forward import walk_forward_predictions

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import optuna
except Exception:
    optuna = None


EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
LEADERBOARD_PATH = OUTPUT_DIR / "experiment_leaderboard.csv"
REPORT_PATH = OUTPUT_DIR / "auto_improve_report.md"
DIAGNOSTICS_PATH = DIAGNOSTICS_DIR / "auto_improve_diagnostics.md"
STUDY_BEST_PARAMS_PATH = OUTPUT_DIR / "optuna_best_params.json"
STUDY_TRIALS_PATH = OUTPUT_DIR / "optuna_trials.csv"
STUDY_SUMMARY_PATH = OUTPUT_DIR / "optuna_summary.md"
SIGNAL_LEADERBOARD_PATH = OUTPUT_DIR / "signal_leaderboard.csv"


def _json_dump(path: Path, obj: Any) -> None:
    """将对象序列化为 UTF-8 JSON 文件。"""
    path.write_text(json.dumps(write_json_ready(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_dirs() -> None:
    """确保自动实验相关目录存在。"""
    for path in [OUTPUT_DIR, PREDICTION_DIR, METRICS_DIR, DIAGNOSTICS_DIR, EXPERIMENTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _feature_top_k_label(top_k: int | None) -> str:
    """将 top_k 配置转换为稳定标签。"""
    return "all" if top_k is None else str(top_k)


def _safe_float(value: Any) -> float | None:
    """将对象安全转为有限浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _failure_reason(exc: BaseException) -> str:
    """将异常归类为便于统计的失败原因。"""
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if "no predictions" in text:
        return "missing_predictions"
    if "valueerror" in text and "horizon" in text:
        return "invalid_horizon"
    if "memory" in text:
        return "memory_error"
    if "lightgbm" in text:
        return "lightgbm_error"
    if "xgboost" in text:
        return "xgboost_error"
    if "prepare_strategy_frame" in text:
        return "frame_build_error"
    return exc.__class__.__name__


def _load_base_artifacts() -> tuple[pd.DataFrame, list[str], list[ModelSpec], list[dict[str, str]]]:
    """加载底层行情、特征和可用模型。"""
    daily, m5 = load_data()
    intraday = build_intraday_features(m5)
    feature_df = build_daily_features(daily, intraday)
    feature_cols = get_feature_columns(feature_df)
    model_specs, skipped_models = build_model_specs()
    return feature_df, feature_cols, model_specs, skipped_models


def _mode_settings(args: argparse.Namespace) -> dict[str, Any]:
    """根据 quick/full 返回默认搜索配置。"""
    if args.full:
        return {
            "feature_top_k_candidates": [50, 80],
            "horizons": [3, 5, 10, 20],
            "model_names": ["HistGradientBoosting", "RandomForest", "ExtraTrees", "LightGBM"],
            "families": [
                "ml_big_up_index_enhancement",
                "ml_bull_full_participation_enhancement",
                "ml_conservative_full_participation_enhancement",
                "ml_ultra_conservative_full_participation_enhancement",
                "ml_direct_signal_timing",
                "ml_big_up_plus_115",
                "ml_big_up_plus_120",
            ],
            "target_labels": [
                "big_up_label",
                "big_down_label",
                "avoid_loss_label",
                "reduce_position_worth_label",
            ],
            "max_runs": args.max_runs or 32,
            "warmup_runs": 4,
            "cache_test_predictions": True,
            "max_signal_candidates_per_top_k": 8,
        }
    return {
        "feature_top_k_candidates": [50],
        "horizons": [5, 10],
        "model_names": ["HistGradientBoosting", "ExtraTrees", "LightGBM"],
        "families": [
            "ml_big_up_index_enhancement",
            "ml_bull_full_participation_enhancement",
            "ml_conservative_full_participation_enhancement",
            "ml_ultra_conservative_full_participation_enhancement",
            "ml_big_up_plus_115",
        ],
        "target_labels": [
            "big_down_label",
            "reduce_position_worth_label",
        ],
        "max_runs": args.max_runs or 4,
        "warmup_runs": 2,
        "cache_test_predictions": True,
        "max_signal_candidates_per_top_k": 5,
    }


def _family_bounds(family: StrategyFamily) -> dict[str, tuple[float, float]]:
    """返回不同策略族的连续参数边界，供 Optuna 和随机降级统一使用。"""
    if family == "ml_bull_full_participation_enhancement":
        return {
            "big_up_prob_threshold": (0.52, 0.62),
            "big_down_prob_threshold": (0.62, 0.78),
            "tail_score_threshold": (0.00, 0.05),
            "mild_cut_exposure": (0.99, 0.999),
            "defensive_cut_exposure": (0.985, 0.995),
            "severe_defensive_exposure": (0.96, 0.985),
            "no_trade_band": (0.02, 0.10),
            "high_volatility_multiplier": (1.10, 1.35),
            "drawdown_threshold": (0.08, 0.14),
            "min_strong_up_position": (0.985, 0.997),
            "opportunity_cost_buffer": (0.0010, 0.0028),
        }
    if family == "ml_ultra_conservative_full_participation_enhancement":
        return {
            "big_up_prob_threshold": (0.54, 0.63),
            "big_down_prob_threshold": (0.60, 0.75),
            "tail_score_threshold": (0.00, 0.06),
            "mild_cut_exposure": (0.975, 0.995),
            "defensive_cut_exposure": (0.95, 0.98),
            "severe_defensive_exposure": (0.92, 0.96),
            "no_trade_band": (0.02, 0.08),
            "high_volatility_multiplier": (1.10, 1.35),
            "drawdown_threshold": (0.08, 0.13),
            "min_strong_up_position": (0.95, 0.985),
            "opportunity_cost_buffer": (0.0008, 0.0022),
        }
    if family == "ml_conservative_full_participation_enhancement":
        return {
            "big_up_prob_threshold": (0.53, 0.63),
            "big_down_prob_threshold": (0.57, 0.72),
            "tail_score_threshold": (0.00, 0.05),
            "mild_cut_exposure": (0.965, 0.99),
            "defensive_cut_exposure": (0.94, 0.97),
            "severe_defensive_exposure": (0.90, 0.95),
            "no_trade_band": (0.015, 0.06),
            "high_volatility_multiplier": (1.08, 1.30),
            "drawdown_threshold": (0.07, 0.12),
            "min_strong_up_position": (0.95, 0.98),
            "opportunity_cost_buffer": (0.0008, 0.0022),
        }
    return {
        "big_up_prob_threshold": (0.53, 0.64),
        "big_down_prob_threshold": (0.55, 0.70),
        "tail_score_threshold": (0.00, 0.05),
        "mild_cut_exposure": (0.95, 0.985),
        "defensive_cut_exposure": (0.88, 0.96),
        "severe_defensive_exposure": (0.82, 0.92),
        "no_trade_band": (0.01, 0.05),
        "high_volatility_multiplier": (1.08, 1.30),
        "drawdown_threshold": (0.06, 0.11),
        "min_strong_up_position": (0.95, 0.98),
        "opportunity_cost_buffer": (0.0008, 0.0025),
    }


def _round_param(name: str, value: float) -> float:
    """按参数类型做离散化，兼顾可解释性和搜索效率。"""
    if name in {"big_up_prob_threshold", "big_down_prob_threshold", "mild_cut_exposure", "defensive_cut_exposure", "severe_defensive_exposure", "min_strong_up_position"}:
        return round(value, 3)
    if name in {"tail_score_threshold", "no_trade_band", "opportunity_cost_buffer"}:
        return round(value, 4)
    return round(value, 3)


class PredictionCache:
    """按需缓存不同 top_k、不同时间窗的 walk-forward 预测结果。"""

    def __init__(
        self,
        feature_df: pd.DataFrame,
        feature_cols: list[str],
        model_specs: list[ModelSpec],
        horizons: list[int],
        cache_test_predictions: bool = True,
    ) -> None:
        """初始化预测缓存管理器。"""
        self.feature_df = feature_df
        self.feature_cols = feature_cols
        self.model_specs = model_specs
        self.horizons = horizons
        self.cache_test_predictions = cache_test_predictions
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get(self, top_k: int | None, period_name: str) -> pd.DataFrame:
        """获取某个 top_k 与某个验证折/测试期的预测，不存在则即时生成。"""
        top_k_label = _feature_top_k_label(top_k)
        key = (top_k_label, period_name)
        should_cache = period_name != "test" or self.cache_test_predictions
        if should_cache and key in self._cache:
            return self._cache[key]

        if period_name == "test":
            start, end = TEST_START, TEST_END
        else:
            fold_map = {name: (start, end) for name, start, end in VALIDATION_FOLDS}
            if period_name not in fold_map:
                raise KeyError(f"Unknown period: {period_name}")
            start, end = fold_map[period_name]

        preds = walk_forward_predictions(
            self.feature_df,
            self.feature_cols,
            self.model_specs,
            start,
            end,
            RETRAIN_EVERY,
            min_train_samples=MIN_TRAIN_SAMPLES,
            label=period_name,
            feature_top_k=top_k,
            horizons=self.horizons,
        )
        preds["validation_fold"] = period_name
        if should_cache:
            self._cache[key] = preds
        return preds

    def warmup(self, top_k_candidates: list[int | None], periods: list[str]) -> None:
        """按需预热常用预测缓存，减少首轮试验等待。"""
        for top_k in top_k_candidates:
            for period_name in periods:
                self.get(top_k, period_name)


def _build_signal_leaderboard(
    prediction_cache: PredictionCache,
    top_k_candidates: list[int | None],
    max_signal_candidates_per_top_k: int,
) -> pd.DataFrame:
    """基于多个验证折的预测指标预筛选信号候选，避免大量无效策略试验。"""
    fold_rows: list[pd.DataFrame] = []
    for top_k in top_k_candidates:
        top_k_label = _feature_top_k_label(top_k)
        for fold_name, _, _ in VALIDATION_FOLDS:
            preds = prediction_cache.get(top_k, fold_name)
            metrics = compute_model_metrics(preds)
            if metrics.empty:
                continue
            metrics = metrics.copy()
            metrics["validation_fold"] = fold_name
            metrics["feature_top_k"] = top_k_label
            fold_rows.append(metrics)

    if not fold_rows:
        raise RuntimeError("无法生成 signal leaderboard，因为验证折预测指标为空。")

    fold_metrics = pd.concat(fold_rows, ignore_index=True)
    aggregated = (
        fold_metrics.groupby(["feature_top_k", "model_name", "horizon"], as_index=False)
        .agg(
            avg_selection_score=("selection_score", "mean"),
            avg_clean_direction_auc=("clean_direction_AUC", "mean"),
            avg_big_up_auc=("big_up_AUC", "mean"),
            avg_big_down_auc=("big_down_AUC", "mean"),
            avg_trade_auc=("trade_AUC", "mean"),
            avg_return_correlation=("return_correlation", "mean"),
            selection_score_std=("selection_score", "std"),
            fold_count=("validation_fold", "nunique"),
        )
    )
    aggregated["selection_score_std"] = aggregated["selection_score_std"].fillna(0.0)
    aggregated["signal_robustness"] = (
        aggregated["avg_selection_score"].fillna(0.0)
        - 0.35 * aggregated["selection_score_std"].fillna(0.0)
        + 0.08 * aggregated["avg_big_up_auc"].fillna(0.5)
        + 0.06 * aggregated["avg_trade_auc"].fillna(0.5)
    )
    aggregated = aggregated.sort_values(
        ["feature_top_k", "signal_robustness", "avg_selection_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    selected = (
        aggregated.groupby("feature_top_k", group_keys=False)
        .head(max_signal_candidates_per_top_k)
        .reset_index(drop=True)
    )
    selected["candidate_id"] = selected.index.astype(str)
    selected.to_csv(SIGNAL_LEADERBOARD_PATH, index=False)
    return selected


def _signal_candidates_by_top_k(signal_leaderboard: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """将信号候选按 top_k 聚合，供搜索器快速采样。"""
    out: dict[str, list[dict[str, Any]]] = {}
    for top_k_label, grp in signal_leaderboard.groupby("feature_top_k", sort=False):
        out[str(top_k_label)] = grp.to_dict("records")
    return out


def _plot_equity_curve(daily_backtest: pd.DataFrame, output_path: Path) -> None:
    """绘制验证折与测试期的资金曲线。"""
    plt.figure(figsize=(12, 6))
    for period, grp in daily_backtest.groupby("dataset_period", sort=False):
        plt.plot(grp["date"], grp["equity"], label=f"strategy-{period}", linewidth=1.6)
        plt.plot(grp["date"], grp["buy_hold_equity"], label=f"buy_hold-{period}", linewidth=1.2, linestyle="--")
    plt.title("Equity Curve by Validation/Test Period")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _backtest_single_period(
    feature_df: pd.DataFrame,
    predictions: pd.DataFrame,
    params: StrategyParams,
    family: StrategyFamily,
    period_name: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """对单个验证折或测试期执行回测。"""
    frame = prepare_strategy_frame(feature_df, predictions, params)
    position, signal = build_strategy_positions(frame, params, family)
    strategy_daily = backtest_position_frame(frame, position, family, signal=signal, initial_position=1.0)
    combined = pd.concat([build_buy_hold_frame(frame), strategy_daily], ignore_index=True)
    metrics = compute_strategy_metrics(combined)
    row = metrics.loc[metrics["strategy"] == family].iloc[0].copy()
    row["dataset_period"] = period_name
    strategy_daily["dataset_period"] = period_name
    return strategy_daily, row


def _markdown_table(df: pd.DataFrame) -> str:
    """将 DataFrame 渲染为简单 Markdown 表格。"""
    if df.empty:
        return "No rows."
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _experiment_summary_markdown(
    config: dict[str, Any],
    validation_summary: dict[str, Any],
    fold_metrics: pd.DataFrame,
    test_metrics: dict[str, Any] | None,
    error: str | None,
) -> str:
    """生成单实验摘要。"""
    lines = [
        "# 实验总结",
        "",
        "## 配置",
        f"- model: {config['model_name']}",
        f"- horizon: {config['horizon']}",
        f"- feature_top_k: {config['feature_top_k']}",
        f"- target_label: {config['decision_target_label']}",
        f"- strategy_family: {config['family']}",
        f"- search_engine: {config.get('search_engine')}",
        "",
        "## 验证期最优依据",
        f"- robust_score: {validation_summary.get('robust_score')}",
        f"- avg_excess_return_vs_buy_hold: {validation_summary.get('avg_excess_return_vs_buy_hold')}",
        f"- positive_fold_ratio: {validation_summary.get('positive_fold_ratio')}",
        f"- avg_missed_upside: {validation_summary.get('avg_missed_upside')}",
        f"- avg_avoided_downside: {validation_summary.get('avg_avoided_downside')}",
        "",
    ]
    if test_metrics is not None:
        lines.extend(
            [
                "## 测试期最终表现",
                f"- test_total_return: {test_metrics.get('total_return')}",
                f"- test_excess_return_vs_buy_hold: {test_metrics.get('excess_return_vs_buy_hold')}",
                f"- test_max_drawdown: {test_metrics.get('max_drawdown')}",
                f"- test_sharpe: {test_metrics.get('sharpe')}",
                "",
                "## 是否跑赢买入持有",
                "- 是" if float(test_metrics.get("excess_return_vs_buy_hold", -999) or -999) > 0 else "- 否",
                "",
            ]
        )
    if error:
        lines.extend(["## 错误", f"```text\n{error}\n```", ""])
    if not fold_metrics.empty:
        lines.append("## 验证折表现")
        lines.append(_markdown_table(fold_metrics))
        lines.append("")
    if test_metrics is not None and float(test_metrics.get("excess_return_vs_buy_hold", -999) or -999) <= 0:
        lines.extend(
            [
                "## 诚实分析",
                "- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。",
                "- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。",
                "- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。",
            ]
        )
    return "\n".join(lines)


def _evaluate_experiment(
    run_id: str,
    exp_idx: int,
    feature_df: pd.DataFrame,
    prediction_cache: PredictionCache,
    params: StrategyParams,
    family: StrategyFamily,
    feature_top_k: int | None,
    search_engine: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行单次实验，并始终落盘结果或错误信息。"""
    exp_dir = EXPERIMENTS_DIR / f"{run_id}_exp_{exp_idx:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    top_k_label = _feature_top_k_label(feature_top_k)
    config = {
        "experiment_id": exp_dir.name,
        "family": family,
        "feature_top_k": top_k_label,
        "search_engine": search_engine,
        **asdict(params),
    }
    if metadata:
        config.update(metadata)
    _json_dump(exp_dir / "config.json", config)

    fold_rows: list[dict[str, Any]] = []
    validation_daily_frames: list[pd.DataFrame] = []
    try:
        for fold_name, _, _ in VALIDATION_FOLDS:
            daily_backtest, metrics_row = _backtest_single_period(
                feature_df,
                prediction_cache.get(feature_top_k, fold_name),
                params,
                family,
                fold_name,
            )
            validation_daily_frames.append(daily_backtest)
            fold_rows.append(metrics_row.to_dict())
        fold_metrics = pd.DataFrame(fold_rows)
        robust_details = robust_score_details_from_fold_metrics(fold_metrics)
        validation_summary = {
            "robust_score": robust_details["robust_score"],
            "avg_excess_return_vs_buy_hold": robust_details["avg_excess_return"],
            "avg_sharpe": robust_details["avg_sharpe"],
            "avg_calmar": robust_details["avg_calmar"],
            "avg_turnover": robust_details["avg_turnover"],
            "avg_max_drawdown": robust_details["avg_max_drawdown"],
            "avg_position": robust_details["avg_position"],
            "avg_missed_upside": robust_details["avg_missed_upside"],
            "avg_avoided_downside": robust_details["avg_avoided_downside"],
            "positive_fold_ratio": robust_details["positive_fold_ratio"],
            "excess_stability": robust_details["excess_stability"],
            "excess_std": robust_details["excess_std"],
            "fold_count": len(fold_metrics),
        }

        test_daily, test_row = _backtest_single_period(
            feature_df,
            prediction_cache.get(feature_top_k, "test"),
            params,
            family,
            "test",
        )
        test_metrics = test_row.to_dict()
        pd.DataFrame([validation_summary]).to_csv(exp_dir / "validation_metrics.csv", index=False)
        fold_metrics.to_csv(exp_dir / "fold_metrics.csv", index=False)
        daily_backtest = pd.concat(validation_daily_frames + [test_daily], ignore_index=True)
        daily_backtest.to_csv(exp_dir / "daily_backtest.csv", index=False)
        _plot_equity_curve(daily_backtest, exp_dir / "equity_curve.png")
        (exp_dir / "summary.md").write_text(
            _experiment_summary_markdown(config, validation_summary, fold_metrics, test_metrics, None),
            encoding="utf-8",
        )
        return {
            **config,
            **validation_summary,
            "status": "ok",
            "failure_reason": None,
            "test_total_return": test_metrics.get("total_return"),
            "test_excess_return_vs_buy_hold": test_metrics.get("excess_return_vs_buy_hold"),
            "test_max_drawdown": test_metrics.get("max_drawdown"),
            "test_sharpe": test_metrics.get("sharpe"),
            "artifact_dir": str(exp_dir),
            "error": None,
        }
    except Exception as exc:
        error_text = traceback.format_exc()
        failure_reason = _failure_reason(exc)
        pd.DataFrame([{"error": error_text, "failure_reason": failure_reason}]).to_csv(exp_dir / "validation_metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(exp_dir / "fold_metrics.csv", index=False)
        pd.DataFrame().to_csv(exp_dir / "daily_backtest.csv", index=False)
        (exp_dir / "summary.md").write_text(
            _experiment_summary_markdown(config, {}, pd.DataFrame(fold_rows), None, error_text),
            encoding="utf-8",
        )
        return {
            **config,
            "status": "failed",
            "failure_reason": failure_reason,
            "robust_score": -1e9,
            "avg_excess_return_vs_buy_hold": None,
            "positive_fold_ratio": None,
            "avg_missed_upside": None,
            "avg_avoided_downside": None,
            "test_total_return": None,
            "test_excess_return_vs_buy_hold": None,
            "test_max_drawdown": None,
            "test_sharpe": None,
            "artifact_dir": str(exp_dir),
            "error": error_text,
        }


def _build_strategy_params_from_mapping(
    family: StrategyFamily,
    model_name: str,
    horizon: int,
    decision_target_label: str,
    mapping: dict[str, float],
) -> StrategyParams:
    """将搜索器产出的参数映射成策略参数对象。"""
    return StrategyParams(
        model_name=model_name,
        horizon=horizon,
        decision_target_label=decision_target_label,
        big_up_prob_threshold=_round_param("big_up_prob_threshold", mapping["big_up_prob_threshold"]),
        big_down_prob_threshold=_round_param("big_down_prob_threshold", mapping["big_down_prob_threshold"]),
        tail_score_threshold=_round_param("tail_score_threshold", mapping["tail_score_threshold"]),
        mild_cut_exposure=_round_param("mild_cut_exposure", mapping["mild_cut_exposure"]),
        defensive_cut_exposure=_round_param("defensive_cut_exposure", mapping["defensive_cut_exposure"]),
        severe_defensive_exposure=_round_param("severe_defensive_exposure", mapping["severe_defensive_exposure"]),
        no_trade_band=_round_param("no_trade_band", mapping["no_trade_band"]),
        high_volatility_multiplier=_round_param("high_volatility_multiplier", mapping["high_volatility_multiplier"]),
        drawdown_threshold=_round_param("drawdown_threshold", mapping["drawdown_threshold"]),
        opportunity_cost_buffer=_round_param("opportunity_cost_buffer", mapping["opportunity_cost_buffer"]),
        min_strong_up_position=_round_param("min_strong_up_position", mapping["min_strong_up_position"]),
    )


def _sample_random_mapping(rng: random.Random, bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    """为随机降级搜索采样一组连续参数。"""
    return {name: rng.uniform(left, right) for name, (left, right) in bounds.items()}


def _study_trial_to_params(
    trial: Any,
    settings: dict[str, Any],
    signal_candidates: list[dict[str, Any]],
) -> tuple[StrategyFamily, int | None, str, int, str, StrategyParams]:
    """使用 Optuna trial 采样实验配置和策略参数。"""
    family = trial.suggest_categorical("family", settings["families"])
    candidate_labels = [item["candidate_id"] for item in signal_candidates]
    candidate_id = trial.suggest_categorical("signal_candidate_id", candidate_labels)
    selected = next(item for item in signal_candidates if item["candidate_id"] == candidate_id)
    feature_top_k_raw = selected["feature_top_k"]
    feature_top_k = None if str(feature_top_k_raw) == "all" else int(feature_top_k_raw)
    model_name = str(selected["model_name"])
    horizon = int(selected["horizon"])
    decision_target_label = trial.suggest_categorical("decision_target_label", settings["target_labels"])
    bounds = _family_bounds(family)
    mapping = {
        name: trial.suggest_float(name, left, right)
        for name, (left, right) in bounds.items()
    }
    params = _build_strategy_params_from_mapping(family, model_name, horizon, decision_target_label, mapping)
    return family, feature_top_k, model_name, horizon, decision_target_label, params


def _write_study_outputs(results_df: pd.DataFrame, study: Any | None, search_engine: str) -> None:
    """落盘搜索器摘要、最优参数和 trial 历史。"""
    ok = results_df.loc[results_df["status"] == "ok"].copy()
    best = ok.sort_values(["robust_score", "test_excess_return_vs_buy_hold"], ascending=[False, False]).iloc[0].to_dict() if not ok.empty else None
    if best is not None:
        _json_dump(STUDY_BEST_PARAMS_PATH, {"search_engine": search_engine, "best_experiment": best})
    else:
        _json_dump(STUDY_BEST_PARAMS_PATH, {"search_engine": search_engine, "best_experiment": None})

    if study is not None and optuna is not None:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(STUDY_TRIALS_PATH, index=False)
        summary = [
            "# Optuna Summary",
            "",
            f"- search_engine: {search_engine}",
            f"- trial_count: {len(trials_df)}",
            f"- best_value: {study.best_value if len(study.trials) else 'NA'}",
        ]
        if best is not None:
            summary.extend(
                [
                    f"- best_experiment_id: {best['experiment_id']}",
                    f"- best_validation_robust_score: {best['robust_score']}",
                    f"- best_test_excess_return_vs_buy_hold: {best['test_excess_return_vs_buy_hold']}",
                ]
            )
        STUDY_SUMMARY_PATH.write_text("\n".join(summary), encoding="utf-8")
    else:
        trial_cols = [
            col
            for col in results_df.columns
            if col in {"experiment_id", "family", "feature_top_k", "model_name", "horizon", "decision_target_label", "robust_score", "status", "failure_reason"}
        ]
        results_df[trial_cols].to_csv(STUDY_TRIALS_PATH, index=False)
        STUDY_SUMMARY_PATH.write_text(
            "\n".join(
                [
                    "# Search Summary",
                    "",
                    f"- search_engine: {search_engine}",
                    "- optuna_unavailable_or_disabled: true",
                    f"- trial_count: {len(results_df)}",
                ]
            ),
            encoding="utf-8",
        )


def _write_reports(
    results: pd.DataFrame,
    skipped_models: list[dict[str, str]],
    settings: dict[str, Any],
    search_engine: str,
    signal_leaderboard: pd.DataFrame | None = None,
) -> None:
    """输出排行榜、总报告和诊断。"""
    results = results.sort_values(["robust_score", "test_excess_return_vs_buy_hold"], ascending=[False, False], na_position="last")
    results.to_csv(LEADERBOARD_PATH, index=False)
    ok = results.loc[results["status"] == "ok"].copy()
    best = ok.iloc[0].to_dict() if not ok.empty else None

    report_lines = [
        "# Auto Improve Report",
        "",
        "## 搜索设置",
        f"- search_engine: {search_engine}",
        f"- validation_folds: {', '.join(name for name, _, _ in VALIDATION_FOLDS)}",
        f"- test_period: {TEST_START.date()} to {TEST_END.date()}",
        f"- max_runs: {settings['max_runs']}",
        f"- feature_top_k_candidates: {settings['feature_top_k_candidates']}",
        f"- families: {settings['families']}",
        f"- target_labels: {settings['target_labels']}",
        "",
        "## 可选模型依赖",
        f"- skipped_optional_models: {json.dumps(skipped_models, ensure_ascii=False)}",
        "",
    ]
    if signal_leaderboard is not None and not signal_leaderboard.empty:
        report_lines.extend(
            [
                "## 验证期信号预筛选",
                f"- signal_leaderboard_path: {SIGNAL_LEADERBOARD_PATH}",
                f"- candidate_count: {len(signal_leaderboard)}",
                f"- top_candidate: {signal_leaderboard.iloc[0]['model_name']} / {int(signal_leaderboard.iloc[0]['horizon'])}D / top_k={signal_leaderboard.iloc[0]['feature_top_k']}",
                "",
            ]
        )
    if best is not None:
        report_lines.extend(
            [
                "## 验证期最优",
                f"- experiment_id: {best['experiment_id']}",
                f"- model/horizon: {best['model_name']} / {best['horizon']}D",
                f"- feature_top_k: {best['feature_top_k']}",
                f"- target_label: {best['decision_target_label']}",
                f"- strategy_family: {best['family']}",
                f"- robust_score: {best['robust_score']}",
                f"- avg_excess_return_vs_buy_hold: {best.get('avg_excess_return_vs_buy_hold')}",
                f"- avg_missed_upside: {best.get('avg_missed_upside')}",
                f"- avg_avoided_downside: {best.get('avg_avoided_downside')}",
                "",
                "## 测试期最终表现",
                f"- test_total_return: {best['test_total_return']}",
                f"- test_excess_return_vs_buy_hold: {best['test_excess_return_vs_buy_hold']}",
                f"- test_max_drawdown: {best['test_max_drawdown']}",
                f"- test_sharpe: {best['test_sharpe']}",
                "",
                "## 是否跑赢买入持有",
                "- 是" if (_safe_float(best.get("test_excess_return_vs_buy_hold")) or -999) > 0 else "- 否",
                "",
            ]
        )
        if (_safe_float(best.get("test_excess_return_vs_buy_hold")) or -999) <= 0:
            report_lines.extend(
                [
                    "## 诚实分析",
                    "- 若 robust_score 不低但测试没赢，说明验证期学到的防守模式在测试单边上涨阶段机会成本偏高。",
                    "- 若 avg_missed_upside 明显偏高，主要问题在仓位层；若验证折本身正超额比例偏低，主要问题在信号层。",
                    "- 测试期结果只用于最终汇报，不参与参数搜索、特征选择和 robust_score 计算。",
                    "",
                ]
            )
    else:
        report_lines.extend(["## 当前最好结果", "- 没有成功实验。", ""])
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    failed = results.loc[results["status"] == "failed"].copy()
    diag_lines = [
        "# Auto Improve Diagnostics",
        "",
        "## 概览",
        f"- total_runs: {len(results)}",
        f"- success_runs: {int((results['status'] == 'ok').sum())}",
        f"- failed_runs: {len(failed)}",
        "",
        "## 防泄漏说明",
        "- 多折验证仅使用 2023H2、2024H1、2024H2。",
        f"- 测试期固定为 {TEST_START.date()} 到 {TEST_END.date()}。",
        "- robust_score、模型选择、参数搜索、特征选择均不使用测试期信息。",
        "- 每个 walk-forward chunk 都只使用 target_end_date < chunk_start 的训练样本。",
        "",
    ]
    if not results.empty:
        model_diag = (
            results.groupby(["model_name", "decision_target_label", "family"], dropna=False)
            .agg(
                run_count=("experiment_id", "count"),
                success_count=("status", lambda s: int((s == "ok").sum())),
                avg_robust_score=("robust_score", "mean"),
            )
            .reset_index()
            .sort_values(["avg_robust_score", "run_count"], ascending=[False, False])
        )
        diag_lines.extend(["## 组合有效性统计", _markdown_table(model_diag.head(12)), ""])
    if not failed.empty:
        failure_counts = failed["failure_reason"].fillna("unknown").value_counts().rename_axis("failure_reason").reset_index(name="count")
        diag_lines.extend(["## 失败原因统计", _markdown_table(failure_counts), ""])
    if skipped_models:
        diag_lines.append("## 跳过的可选模型")
        for item in skipped_models:
            diag_lines.append(f"- {item}")
        diag_lines.append("")
    if signal_leaderboard is not None and not signal_leaderboard.empty:
        diag_lines.extend(["## 信号预筛选前列候选", _markdown_table(signal_leaderboard.head(12)), ""])
    DIAGNOSTICS_PATH.write_text("\n".join(diag_lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="自动化改进机器学习指数增强策略")
    parser.add_argument("--quick", action="store_true", help="运行快速回归搜索")
    parser.add_argument("--full", action="store_true", help="运行更完整的搜索")
    parser.add_argument("--max-runs", type=int, default=None, help="最大实验次数")
    parser.add_argument("--timeout", type=int, default=None, help="Optuna 搜索超时秒数")
    parser.add_argument("--seed", type=int, default=42, help="搜索随机种子")
    parser.add_argument("--disable-optuna", action="store_true", help="强制禁用 Optuna，退回随机搜索")
    parser.add_argument("--diagnose-only", action="store_true", help="只重建诊断报告")
    parser.add_argument("--force", action="store_true", help="清理旧实验目录后重跑")
    return parser.parse_args()


def _diagnose_only() -> None:
    """只根据现有 leaderboard 重建报告。"""
    if not LEADERBOARD_PATH.exists():
        raise FileNotFoundError(f"未找到现有 leaderboard: {LEADERBOARD_PATH}")
    results = pd.read_csv(LEADERBOARD_PATH)
    signal_leaderboard = pd.read_csv(SIGNAL_LEADERBOARD_PATH) if SIGNAL_LEADERBOARD_PATH.exists() else None
    _write_reports(
        results,
        [],
        {"max_runs": len(results), "feature_top_k_candidates": [], "families": [], "target_labels": []},
        "diagnose_only",
        signal_leaderboard,
    )


def _run_random_search(
    run_id: str,
    feature_df: pd.DataFrame,
    prediction_cache: PredictionCache,
    settings: dict[str, Any],
    signal_candidates: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """当 Optuna 不可用时，执行统一接口下的随机搜索。"""
    rng = random.Random(seed)
    results: list[dict[str, Any]] = []
    for exp_idx in range(1, settings["max_runs"] + 1):
        family = rng.choice(settings["families"])
        selected = rng.choice(signal_candidates)
        top_k_raw = selected["feature_top_k"]
        top_k = None if str(top_k_raw) == "all" else int(top_k_raw)
        model_name = str(selected["model_name"])
        horizon = int(selected["horizon"])
        target_label = rng.choice(settings["target_labels"])
        mapping = _sample_random_mapping(rng, _family_bounds(family))
        params = _build_strategy_params_from_mapping(family, model_name, horizon, target_label, mapping)
        print(
            f"[{exp_idx}/{settings['max_runs']}] random family={family}, model={model_name}, "
            f"horizon={horizon}, top_k={_feature_top_k_label(top_k)}, target={target_label}"
        )
        results.append(
            _evaluate_experiment(
                run_id,
                exp_idx,
                feature_df,
                prediction_cache,
                params,
                family,
                top_k,
                "random",
                metadata={"seed": seed},
            )
        )
    return results


def _run_optuna_search(
    run_id: str,
    feature_df: pd.DataFrame,
    prediction_cache: PredictionCache,
    settings: dict[str, Any],
    signal_candidates: list[dict[str, Any]],
    seed: int,
    timeout: int | None,
) -> tuple[list[dict[str, Any]], Any]:
    """使用 Optuna 执行统一搜索，并记录每个 trial 的真实产物。"""
    sampler = optuna.samplers.TPESampler(seed=seed) if optuna is not None else None
    study = optuna.create_study(direction="maximize", sampler=sampler)  # type: ignore[union-attr]
    results: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        exp_idx = len(results) + 1
        family, top_k, model_name, horizon, target_label, params = _study_trial_to_params(trial, settings, signal_candidates)
        print(
            f"[{exp_idx}/{settings['max_runs']}] optuna family={family}, model={model_name}, "
            f"horizon={horizon}, top_k={_feature_top_k_label(top_k)}, target={target_label}"
        )
        result = _evaluate_experiment(
            run_id,
            exp_idx,
            feature_df,
            prediction_cache,
            params,
            family,
            top_k,
            "optuna",
            metadata={"trial_number": trial.number, "seed": seed},
        )
        results.append(result)
        trial.set_user_attr("experiment_id", result["experiment_id"])
        trial.set_user_attr("status", result["status"])
        trial.set_user_attr("failure_reason", result.get("failure_reason"))
        trial.set_user_attr("test_excess_return_vs_buy_hold", result.get("test_excess_return_vs_buy_hold"))
        return float(result["robust_score"])

    study.optimize(objective, n_trials=settings["max_runs"], timeout=timeout)
    return results, study


def main() -> None:
    """运行自动改进搜索，优先使用 Optuna，必要时降级为随机搜索。"""
    args = _parse_args()
    _ensure_dirs()
    if args.diagnose_only:
        _diagnose_only()
        print(f"Diagnostics regenerated at: {DIAGNOSTICS_PATH}")
        return

    settings = _mode_settings(args)
    if args.force and EXPERIMENTS_DIR.exists():
        shutil.rmtree(EXPERIMENTS_DIR)
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data, features, and models for auto improvement...")
    feature_df, feature_cols, model_specs, skipped_models = _load_base_artifacts()
    print(f"Feature count: {len(feature_cols)}")
    print(f"Available models: {[spec.name for spec in model_specs]}")
    if skipped_models:
        print(f"Skipped optional models: {skipped_models}")

    selected_model_names = settings["model_names"]
    active_model_specs = (
        [spec for spec in model_specs if spec.name in set(selected_model_names)]
        if selected_model_names is not None
        else model_specs
    )
    model_names = sorted({spec.name for spec in active_model_specs})
    print(f"Active search models: {model_names}")
    print(f"Active search horizons: {settings['horizons']}")

    print("Preparing lazy multi-fold prediction cache...")
    prediction_cache = PredictionCache(
        feature_df,
        feature_cols,
        active_model_specs,
        settings["horizons"],
        cache_test_predictions=settings["cache_test_predictions"],
    )
    warmup_periods = [name for name, _, _ in VALIDATION_FOLDS[: settings["warmup_runs"]]]
    prediction_cache.warmup(settings["feature_top_k_candidates"][:1], warmup_periods)
    print("Building validation-only signal leaderboard...")
    signal_leaderboard = _build_signal_leaderboard(
        prediction_cache,
        settings["feature_top_k_candidates"],
        settings["max_signal_candidates_per_top_k"],
    )
    signal_candidates = signal_leaderboard.to_dict("records")
    print(
        "Top signal candidates: "
        + ", ".join(
            f"{row['model_name']}-{int(row['horizon'])}D-topk{row['feature_top_k']}"
            for row in signal_candidates[: min(5, len(signal_candidates))]
        )
    )

    run_id = datetime.now().strftime("auto_%Y%m%d_%H%M%S")
    use_optuna = optuna is not None and not args.disable_optuna

    if use_optuna:
        print("Running Optuna search...")
        results, study = _run_optuna_search(
            run_id,
            feature_df,
            prediction_cache,
            settings,
            signal_candidates,
            args.seed,
            args.timeout,
        )
        search_engine = "optuna"
    else:
        if optuna is None:
            print("Optuna unavailable, falling back to random search...")
        else:
            print("Optuna disabled, falling back to random search...")
        results = _run_random_search(
            run_id,
            feature_df,
            prediction_cache,
            settings,
            signal_candidates,
            args.seed,
        )
        study = None
        search_engine = "random"

    results_df = pd.DataFrame(results)
    _write_reports(results_df, skipped_models, settings, search_engine, signal_leaderboard)
    _write_study_outputs(results_df, study, search_engine)
    print(f"Leaderboard saved to: {LEADERBOARD_PATH}")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Diagnostics saved to: {DIAGNOSTICS_PATH}")
    print(f"Study best params saved to: {STUDY_BEST_PARAMS_PATH}")
    print(f"Study trials saved to: {STUDY_TRIALS_PATH}")


if __name__ == "__main__":
    main()
