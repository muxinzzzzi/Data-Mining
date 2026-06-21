from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CURRENT_DIR = OUTPUTS_DIR / "current"
ARCHIVE_DIR = OUTPUTS_DIR / "archive_outputs"

CURRENT_SUBDIRS = [
    CURRENT_DIR / "plots",
    CURRENT_DIR / "diagnostics",
    CURRENT_DIR / "attribution",
    CURRENT_DIR / "params",
    CURRENT_DIR / "misc",
]

ARCHIVE_SUBDIRS = [
    ARCHIVE_DIR / "old_runs",
    ARCHIVE_DIR / "obsolete_diagnostics",
    ARCHIVE_DIR / "obsolete_plots",
    ARCHIVE_DIR / "obsolete_predictions",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}

MAIN_RESULT_FILES = {
    "summary.md",
    "final_report_material.md",
    "strategy_metrics_all.csv",
    "strategy_daily_all.csv",
    "final_strategy_comparison.csv",
    "final_strategy_comparison.md",
    "external_market_feature_report.md",
    "model_stability_audit.csv",
    "model_stability_audit.md",
    "model_strategy_utility_comparison.csv",
    "model_strategy_utility_comparison.md",
}

ATTRIBUTION_PATTERNS = [
    "attribution",
    "top_missed_upside",
    "top_successful_defensive",
    "directional_timing",
    "active_return",
]

PARAM_PATTERNS = [
    "tuned_params",
    "_selection",
    "model_selection",
    "signal_selection",
    "feature_top_k_selection",
    "strategy_tuned_params",
    "ablation_params",
    "best_prediction_model",
]

DIAGNOSTIC_PATTERNS = [
    "diagnostic",
    "diagnostics",
    "signal_bucket",
    "signal_quantile",
    "validation_to_test",
    "next_optimization",
    "prediction_summary",
    "model_metrics",
    "factor_effectiveness",
    "feature_missing",
    "label_balance",
    "correlation",
    "calibration",
    "underperformance",
    "exposure",
    "conservative_strategy_comparison",
    "high_big_up_but_weak_return",
    "low_big_up_but_positive_return",
    "stability_aware_upside_score",
]


@dataclass(frozen=True)
class CopyPlan:
    source: Path
    target: Path
    category: str
    unclassified: bool = False


def ensure_directories() -> list[Path]:
    dirs = [CURRENT_DIR, ARCHIVE_DIR, *CURRENT_SUBDIRS, *ARCHIVE_SUBDIRS]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def iter_source_files() -> Iterable[Path]:
    source_dirs = [
        OUTPUTS_DIR,
        OUTPUTS_DIR / "plots",
        OUTPUTS_DIR / "diagnostics",
        OUTPUTS_DIR / "diagnostics" / "plots",
        OUTPUTS_DIR / "data_diagnostics",
        OUTPUTS_DIR / "data_diagnostics" / "plots",
        OUTPUTS_DIR / "metrics",
        OUTPUTS_DIR / "predictions",
    ]
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.iterdir()):
            if not path.is_file():
                continue
            if should_skip(path):
                continue
            yield path


def should_skip(path: Path) -> bool:
    if path.name == ".DS_Store":
        return True
    if path.suffix == ".pyc":
        return True
    try:
        parts = path.relative_to(OUTPUTS_DIR).parts
    except ValueError:
        return True
    return bool(parts and parts[0] in {"current", "archive_outputs"})


def target_name_for_plot(path: Path) -> str:
    rel = path.relative_to(OUTPUTS_DIR)
    if len(rel.parts) >= 3 and rel.parts[-2] == "plots":
        return "__".join(rel.parts[:-1]) + "__" + path.name
    return path.name


def classify(path: Path) -> CopyPlan:
    rel = path.relative_to(OUTPUTS_DIR).as_posix()
    name = path.name
    lower_name = name.lower()
    lower_rel = rel.lower()

    if path.suffix.lower() in IMAGE_EXTENSIONS and "plots/" in lower_rel:
        return CopyPlan(path, CURRENT_DIR / "plots" / target_name_for_plot(path), "plots")

    if path.parent == OUTPUTS_DIR and name in MAIN_RESULT_FILES:
        return CopyPlan(path, CURRENT_DIR / name, "current")

    if rel == "diagnostics/feature_columns.json":
        return CopyPlan(path, CURRENT_DIR / "feature_columns.json", "current")

    if rel == "metrics/best_prediction_model.json":
        return CopyPlan(path, CURRENT_DIR / "best_model_selection.json", "current")

    if any(pattern in lower_name for pattern in ATTRIBUTION_PATTERNS):
        return CopyPlan(path, CURRENT_DIR / "attribution" / name, "attribution")

    if lower_name.endswith(".json") and any(pattern in lower_name for pattern in PARAM_PATTERNS):
        return CopyPlan(path, CURRENT_DIR / "params" / name, "params")

    if any(pattern in lower_name for pattern in DIAGNOSTIC_PATTERNS):
        return CopyPlan(path, CURRENT_DIR / "diagnostics" / name, "diagnostics")

    if rel.startswith("diagnostics/") or rel.startswith("metrics/") or rel.startswith("data_diagnostics/"):
        return CopyPlan(path, CURRENT_DIR / "diagnostics" / name, "diagnostics")

    return CopyPlan(path, CURRENT_DIR / "misc" / name, "misc", unclassified=True)


def copy_plan_items(plans: list[CopyPlan]) -> int:
    copied = 0
    for plan in plans:
        plan.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.source, plan.target)
        copied += 1
    return copied


def read_strategy_metrics() -> list[dict[str, str]]:
    path = OUTPUTS_DIR / "strategy_metrics_all.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_decimal(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "未检测到"
    return f"{value:.{digits}f}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "未检测到"
    return f"{value * 100:.2f}%"


def bool_text(flag: bool | None) -> str:
    if flag is None:
        return "未检测到"
    return "是" if flag else "否"


def row_by_strategy(rows: list[dict[str, str]], strategy: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("strategy") == strategy:
            return row
    return None


def is_benchmark_clone(row: dict[str, str]) -> bool:
    raw = str(row.get("benchmark_clone", "")).strip().lower()
    return raw in {"true", "1", "yes"}


def is_enhanced_exposure(row: dict[str, str]) -> bool:
    strategy = str(row.get("strategy", ""))
    max_position = to_float(row, "maximum_position")
    return "plus_" in strategy or (max_position is not None and max_position > 1.000001)


def best_real_no_leverage_strategy(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates: list[dict[str, str]] = []
    for row in rows:
        strategy = str(row.get("strategy", ""))
        if strategy == "buy_hold" or not strategy.startswith("ml_"):
            continue
        if is_benchmark_clone(row) or is_enhanced_exposure(row):
            continue
        max_position = to_float(row, "maximum_position")
        days_below = to_float(row, "days_below_full_exposure") or 0.0
        avg_gap = to_float(row, "avg_abs_position_gap_from_1") or 0.0
        if max_position is not None and max_position > 1.000001:
            continue
        if days_below <= 0 and avg_gap <= 1e-8:
            continue
        candidates.append(row)
    return max(candidates, key=lambda row: to_float(row, "total_return") or float("-inf"), default=None)


def best_enhanced_exposure_strategy(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if is_enhanced_exposure(row)]
    return max(candidates, key=lambda row: to_float(row, "total_return") or float("-inf"), default=None)


def list_files_for_markdown(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    files = sorted(path for path in directory.iterdir() if path.is_file())
    return [path.name for path in files]


def write_output_index(plans: list[CopyPlan], unclassified: list[CopyPlan]) -> None:
    sections = {
        "current": list_files_for_markdown(CURRENT_DIR),
        "diagnostics": list_files_for_markdown(CURRENT_DIR / "diagnostics"),
        "attribution": list_files_for_markdown(CURRENT_DIR / "attribution"),
        "params": list_files_for_markdown(CURRENT_DIR / "params"),
        "plots": list_files_for_markdown(CURRENT_DIR / "plots"),
        "misc": list_files_for_markdown(CURRENT_DIR / "misc"),
    }
    lines: list[str] = [
        "# Outputs Current Index",
        "",
        "本索引由 `scripts/organize_outputs.py` 生成，只整理文件副本，不修改原始实验结果。",
        "",
        "## 子目录作用",
        "- `outputs/current/`: 当前主线报告、主策略指标和关键配置副本。",
        "- `outputs/current/plots/`: 当前报告和诊断可引用的图表。",
        "- `outputs/current/diagnostics/`: 信号有效性、回撤事件、预测和数据质量诊断。",
        "- `outputs/current/attribution/`: 策略归因、主动收益归因和防守日明细。",
        "- `outputs/current/params/`: 验证期选择出的参数、模型选择和信号选择 JSON。",
        "- `outputs/current/misc/`: 暂未被规则明确分类的输出副本。",
        "- `outputs/archive_outputs/`: 预留给历史版本归档；历史旧输出不作为最终主结果引用。",
        "",
        "## 当前主报告优先看",
        "- `outputs/current/summary.md`",
        "- `outputs/current/final_report_material.md`",
        "- `outputs/current/external_market_feature_report.md`",
        "- `outputs/current/model_stability_audit.md`",
        "- `outputs/current/model_strategy_utility_comparison.md`",
        "- `outputs/current/CURRENT_RESULT_README.md`",
        "",
        "## 当前主结果优先看",
        "- `outputs/current/strategy_metrics_all.csv`",
        "- `outputs/current/strategy_daily_all.csv`",
        "- `outputs/current/best_model_selection.json`",
        "- `outputs/current/feature_columns.json`",
    ]
    if (CURRENT_DIR / "final_strategy_comparison.csv").exists():
        lines.append("- `outputs/current/final_strategy_comparison.csv`")
    else:
        lines.append("- 未检测到 `final_strategy_comparison.csv`，建议后续单独生成。")
    if (CURRENT_DIR / "final_strategy_comparison.md").exists():
        lines.append("- `outputs/current/final_strategy_comparison.md`")
    else:
        lines.append("- 未检测到 `final_strategy_comparison.md`，建议后续单独生成。")

    lines.extend(
        [
            "",
            "## 分类文件清单",
            "",
            "### current/",
        ]
    )
    for filename in sections["current"]:
        lines.append(f"- `{filename}`")
    for section in ["diagnostics", "attribution", "params", "plots", "misc"]:
        lines.extend(["", f"### current/{section}/"])
        if sections[section]:
            lines.extend(f"- `{filename}`" for filename in sections[section])
        else:
            lines.append("- 空")

    lines.extend(
        [
            "",
            "## 未分类文件",
            f"- 未分类文件数量: {len(unclassified)}",
        ]
    )
    if unclassified:
        lines.extend(f"- `{plan.source.relative_to(OUTPUTS_DIR).as_posix()}` -> `current/misc/{plan.target.name}`" for plan in unclassified)
    else:
        lines.append("- 无")

    lines.extend(
        [
            "",
            "## 说明",
            "- `outputs/current/` 是当前主线阅读入口。",
            "- `outputs/archive_outputs/` 和散落在旧 `outputs/` 路径下的历史文件仅作追溯，不作为最终主结果。",
            "- 本脚本不会删除、重算或改写原始实验结果数值。",
            f"- 本次复制文件数量: {len(plans)}",
        ]
    )
    (CURRENT_DIR / "OUTPUT_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_current_result_readme() -> None:
    rows = read_strategy_metrics()
    buy_hold = row_by_strategy(rows, "buy_hold")
    best_real = best_real_no_leverage_strategy(rows)
    best_enhanced = best_enhanced_exposure_strategy(rows)
    early_stress = row_by_strategy(rows, "ml_early_stress_risk_budget")
    risk_budget = row_by_strategy(rows, "ml_risk_budget_enhancement")

    best_real_name = best_real.get("strategy") if best_real else "未检测到"
    best_real_excess = to_float(best_real, "excess_return_vs_buy_hold") if best_real else None
    best_real_outperforms = best_real_excess is not None and best_real_excess > 0
    early_drawdown_improvement = None
    if early_stress and buy_hold:
        early_mdd = to_float(early_stress, "max_drawdown")
        buy_hold_mdd = to_float(buy_hold, "max_drawdown")
        if early_mdd is not None and buy_hold_mdd is not None:
            early_drawdown_improvement = early_mdd - buy_hold_mdd

    final_comparison_csv = CURRENT_DIR / "final_strategy_comparison.csv"
    final_comparison_md = CURRENT_DIR / "final_strategy_comparison.md"

    lines = [
        "# 当前结果说明",
        "",
        "## 项目主线",
        "本项目当前主线是“机器学习预测 + 880823 微盘股指数增强策略”。研究重点是检验 ML 预测信号能否在不使用杠杆、不做空、不修改 buy-and-hold 基准和交易成本的前提下，改善指数增强策略的收益、回撤或解释性。",
        "",
        "## 最重要的结果文件",
        "- `outputs/current/summary.md`: 当前策略结果摘要。",
        "- `outputs/current/final_report_material.md`: 最终报告材料。",
        "- `outputs/current/external_market_feature_report.md`: 外部市场数据特征和消融实验报告。",
        "- `outputs/current/model_stability_audit.md`: 预测层模型稳定性审计。",
        "- `outputs/current/model_strategy_utility_comparison.md`: 模型信号进入策略后的效用对比。",
        "- `outputs/current/strategy_metrics_all.csv`: 全部策略测试期指标。",
        "- `outputs/current/strategy_daily_all.csv`: 全部策略日度净值、仓位和收益。",
        "- `outputs/current/feature_columns.json`: 当前特征列清单副本。",
        "- `outputs/current/best_model_selection.json`: 当前最佳预测模型选择副本。",
        "",
        "## 当前关键结论",
        f"- Buy-and-hold total return: {fmt_decimal(to_float(buy_hold, 'total_return') if buy_hold else None)} ({fmt_pct(to_float(buy_hold, 'total_return') if buy_hold else None)})",
        f"- Buy-and-hold max drawdown: {fmt_decimal(to_float(buy_hold, 'max_drawdown') if buy_hold else None)} ({fmt_pct(to_float(buy_hold, 'max_drawdown') if buy_hold else None)})",
        f"- 当前最好无杠杆真实 ML 策略: `{best_real_name}`",
        f"- 最好无杠杆真实 ML 策略 total return: {fmt_decimal(to_float(best_real, 'total_return') if best_real else None)} ({fmt_pct(to_float(best_real, 'total_return') if best_real else None)})",
        f"- 最好无杠杆真实 ML 策略 excess return: {fmt_decimal(best_real_excess)} ({fmt_pct(best_real_excess)})",
        f"- 当前是否跑赢 buy-and-hold: {bool_text(best_real_outperforms)}",
        f"- Early-stress risk-budget 是否降低最大回撤: {bool_text(early_drawdown_improvement is not None and early_drawdown_improvement > 0)}",
        f"- Early-stress max drawdown improvement: {fmt_decimal(early_drawdown_improvement)} ({fmt_pct(early_drawdown_improvement)})",
    ]

    if best_enhanced:
        lines.extend(
            [
                "",
                "## Enhanced-Exposure 单独报告",
                f"- 当前检测到 enhanced-exposure 策略: `{best_enhanced.get('strategy')}`。",
                "- enhanced-exposure 允许仓位高于 1.00，因此只单独报告，不作为无杠杆主结果。",
                f"- Enhanced-exposure excess return: {fmt_decimal(to_float(best_enhanced, 'excess_return_vs_buy_hold'))} ({fmt_pct(to_float(best_enhanced, 'excess_return_vs_buy_hold'))})",
            ]
        )
    else:
        lines.extend(["", "## Enhanced-Exposure 单独报告", "- 未检测到 enhanced-exposure 策略。"])

    lines.extend(
        [
            "",
            "## 回撤控制相关结论",
            f"- 原始 risk-budget 策略: `{risk_budget.get('strategy') if risk_budget else '未检测到'}`。",
            f"- Early-stress 策略: `{early_stress.get('strategy') if early_stress else '未检测到'}`。",
            "- 当前结果显示 early-stress risk-budget 在最大回撤事件前更早降仓，并小幅降低最大回撤；但它仍未跑赢 buy-and-hold，主要原因仍是强上涨测试期中防守动作带来的 missed upside。",
            "",
            "## 最终报告建议引用",
            "- Markdown: `outputs/current/summary.md`, `outputs/current/final_report_material.md`, `outputs/current/diagnostics/drawdown_event_diagnostics_summary.md`",
            "- CSV: `outputs/current/strategy_metrics_all.csv`, `outputs/current/strategy_daily_all.csv`, `outputs/current/diagnostics/drawdown_event_diagnostics.csv`",
            "- 参数 JSON: `outputs/current/params/early_stress_risk_budget_tuned_params.json`, `outputs/current/params/stability_aware_high_participation_tuned_params.json`, `outputs/current/params/stability_aware_signal_selection.json`",
            "- 图表: `outputs/current/plots/equity_curve_early_stress_risk_budget_vs_buyhold.png`, `outputs/current/plots/drawdown_curve_early_stress_risk_budget_vs_buyhold.png`, `outputs/current/plots/drawdown_event_risk_signal_timeline.png`",
        ]
    )

    if not final_comparison_csv.exists() or not final_comparison_md.exists():
        lines.extend(
            [
                "",
                "## final_strategy_comparison 状态",
                "- 未检测到 `final_strategy_comparison.csv` 或 `final_strategy_comparison.md`。",
                "- 本次整理不会凭空编造该结果，建议后续在研究逻辑中单独生成。",
            ]
        )

    lines.extend(
        [
            "",
            "## 归档说明",
            "- `outputs/current/` 是当前主线入口。",
            "- `outputs/archive_outputs/` 预留给历史版本或旧实验归档。",
            "- 散落在旧 `outputs/` 路径下的文件仍保留原样，不作为最终主结果优先引用。",
        ]
    )
    (CURRENT_DIR / "CURRENT_RESULT_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    created_dirs = ensure_directories()
    plans = [classify(path) for path in iter_source_files()]
    copied_count = copy_plan_items(plans)
    unclassified = [plan for plan in plans if plan.unclassified]
    write_current_result_readme()
    write_output_index(plans, unclassified)

    print("===== OUTPUT ORGANIZATION COMPLETE =====")
    print("Created directories:")
    for directory in created_dirs:
        print(f"- {directory}")
    print(f"Copied files: {copied_count}")
    print(f"Unclassified files: {len(unclassified)}")
    if unclassified:
        print("Unclassified file list:")
        for plan in unclassified:
            print(f"- {plan.source.relative_to(OUTPUTS_DIR).as_posix()} -> {plan.target.relative_to(OUTPUTS_DIR).as_posix()}")
    print(f"Current main report path: {CURRENT_DIR / 'summary.md'}")
    print(f"Current strategy metrics path: {CURRENT_DIR / 'strategy_metrics_all.csv'}")
    print(f"Current plots directory path: {CURRENT_DIR / 'plots'}")
    print("========================================")


if __name__ == "__main__":
    main()
