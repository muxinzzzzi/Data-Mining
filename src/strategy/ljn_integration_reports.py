from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LJN_SECTION_TITLE = "## LJN Integration Experiment"


VALIDATION_COLUMNS = [
    "strategy_name",
    "model_name",
    "horizon",
    "fold_name",
    "fold_start",
    "fold_end",
    "total_return",
    "excess_return_vs_buy_hold",
    "max_drawdown",
    "sharpe",
    "calmar",
    "average_position",
    "total_turnover",
    "transaction_cost",
    "robust_score",
]

AGGREGATE_COLUMNS = [
    "strategy_name",
    "model_name",
    "horizon",
    "mean_validation_excess",
    "min_validation_excess",
    "std_validation_excess",
    "mean_validation_calmar",
    "min_validation_calmar",
    "mean_turnover",
    "selected_by_multifold_validation",
    "test_total_return",
    "test_excess_return_vs_buy_hold",
    "test_max_drawdown",
    "test_sharpe",
    "test_calmar",
    "test_average_position",
    "test_turnover",
    "beats_current_main_strategy",
    "beats_buy_hold",
    "final_recommendation",
]


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(val):
        return "NA"
    return f"{val:.{digits}f}"


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    use = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows).copy()
    if use.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(use.columns) + " |",
        "| " + " | ".join(["---"] * len(use.columns)) + " |",
    ]
    for _, row in use.iterrows():
        values: list[str] = []
        for col in use.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(_fmt(value, 6))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _jsonify_params(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _prepare_validation(validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.copy()
    if "params" in out.columns:
        out["params"] = out["params"].map(_jsonify_params)
    preferred = [
        "config_id",
        *VALIDATION_COLUMNS,
        "selected_by_multifold_validation",
        "params",
        "missed_upside",
        "avoided_downside",
    ]
    cols = [col for col in preferred if col in out.columns] + [col for col in out.columns if col not in preferred]
    return out.loc[:, cols]


def _prepare_aggregate(aggregate: pd.DataFrame) -> pd.DataFrame:
    out = aggregate.copy()
    if "params" in out.columns:
        out["params"] = out["params"].map(_jsonify_params)
    preferred = [
        "config_id",
        *AGGREGATE_COLUMNS,
        "robust_score",
        "mean_validation_position",
        "params",
    ]
    cols = [col for col in preferred if col in out.columns] + [col for col in out.columns if col not in preferred]
    return out.loc[:, cols]


def _upsert_summary_section(summary_path: Path, section: str) -> None:
    existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = f"\n{LJN_SECTION_TITLE}\n"
    inline_marker = f"{LJN_SECTION_TITLE}\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    elif existing.startswith(inline_marker):
        existing = ""
    summary_path.write_text(existing.rstrip() + "\n" + section.strip() + "\n", encoding="utf-8")


def build_ljn_summary_section(
    aggregate: pd.DataFrame,
    current_main_total_return: float,
    current_main_excess: float,
    current_main_max_drawdown: float,
) -> str:
    selected = aggregate.loc[aggregate["selected_by_multifold_validation"].astype(bool)].copy()
    selected_row = selected.iloc[0] if not selected.empty else None
    best_test = aggregate.sort_values("test_total_return", ascending=False, na_position="last").iloc[0] if not aggregate.empty else None
    any_beats_buy_hold = bool((aggregate["test_excess_return_vs_buy_hold"] > 0).any()) if not aggregate.empty else False
    any_beats_main = bool((aggregate["test_total_return"] > current_main_total_return).any()) if not aggregate.empty else False

    selected_beats_main = bool(selected_row is not None and selected_row.get("test_total_return", -np.inf) > current_main_total_return)
    selected_beats_buy_hold = bool(selected_row is not None and selected_row.get("test_excess_return_vs_buy_hold", -np.inf) > 0)
    validation_stable = bool(selected_row is not None and selected_row.get("min_validation_excess", -np.inf) > 0)
    recommend_upgrade = selected_beats_main and selected_beats_buy_hold and validation_stable
    if recommend_upgrade:
        upgrade_text = "Yes; validation-selected candidate is stable and also beats the current main strategy on the one-shot test evaluation."
    else:
        upgrade_text = "No; keep `ml_risk_budget_enhancement` as current main strategy."

    selected_name = selected_row.get("strategy_name", "NA") if selected_row is not None else "NA"
    selected_model = selected_row.get("model_name", "NA") if selected_row is not None else "NA"
    selected_horizon = selected_row.get("horizon", "NA") if selected_row is not None else "NA"
    selected_validation = _fmt(selected_row.get("mean_validation_excess") if selected_row is not None else None)
    selected_min_validation = _fmt(selected_row.get("min_validation_excess") if selected_row is not None else None)
    selected_test = _fmt(selected_row.get("test_total_return") if selected_row is not None else None)
    selected_test_excess = _fmt(selected_row.get("test_excess_return_vs_buy_hold") if selected_row is not None else None)
    selected_test_mdd = _fmt(selected_row.get("test_max_drawdown") if selected_row is not None else None)
    best_test_name = best_test.get("strategy_name", "NA") if best_test is not None else "NA"
    best_test_config = best_test.get("config_id", "NA") if best_test is not None else "NA"
    best_test_return = _fmt(best_test.get("test_total_return") if best_test is not None else None)

    instability_reason = (
        "The selected candidate does not have positive excess in every validation fold."
        if not validation_stable
        else "The selected candidate does not clear the current main strategy on test."
    )
    if recommend_upgrade:
        instability_reason = "Validation and one-shot test both clear the upgrade gate."

    return f"""
{LJN_SECTION_TITLE}
- Direct ljn merge: No. The ljn branch was used only as a candidate idea library.
- Ideas absorbed: multi-fold validation, `max_position_change` position smoothing, and reduce-position-worth auxiliary target.
- Current main strategy remains: `ml_risk_budget_enhancement` / LightGBM / 5D; test total return {_fmt(current_main_total_return)}, excess {_fmt(current_main_excess)}, max drawdown {_fmt(current_main_max_drawdown)}.
- Multi-fold selection rule: validation folds only, using `mean_validation_excess + 0.5 * min_validation_excess + 0.2 * mean_validation_calmar - 0.1 * std_validation_excess - 0.05 * mean_turnover`.
- Validation-selected candidate: `{selected_name}` / `{selected_model}` / {selected_horizon}D; mean validation excess {selected_validation}, min validation excess {selected_min_validation}.
- Validation stability: {"Yes" if validation_stable else "No"}.
- Test result for validation-selected candidate: total return {selected_test}, excess {selected_test_excess}, max drawdown {selected_test_mdd}.
- Any ljn-inspired candidate beats buy-and-hold on test: {"Yes" if any_beats_buy_hold else "No"}.
- Any ljn-inspired candidate beats current main strategy on test: {"Yes" if any_beats_main else "No"}; best test candidate by total return is `{best_test_name}` config `{best_test_config}` at {best_test_return}.
- Upgrade main strategy: {upgrade_text}
- Reason: {instability_reason}
- Ideas worth retaining if not upgraded: multi-fold validation as a stability gate; position smoothing as a turnover/cost control; reduce-position-worth labels as a conservative auxiliary risk signal.
"""


def write_ljn_integration_reports(
    validation: pd.DataFrame,
    aggregate: pd.DataFrame,
    output_dir: Path,
    summary_path: Path,
    current_main_total_return: float,
    current_main_excess: float,
    current_main_max_drawdown: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_out = _prepare_validation(validation)
    aggregate_out = _prepare_aggregate(aggregate)

    validation_out.to_csv(output_dir / "ljn_multifold_validation.csv", index=False)
    aggregate_out.to_csv(output_dir / "ljn_integration_experiment.csv", index=False)

    validation_md = f"""# LJN Multi-Fold Validation

This is a controlled ljn-inspired experiment. It does not directly merge the ljn branch and does not replace the current main strategy.

Robust score:
`mean_validation_excess + 0.5 * min_validation_excess + 0.2 * mean_validation_calmar - 0.1 * std_validation_excess - 0.05 * mean_turnover`

{_markdown_table(validation_out.sort_values(["selected_by_multifold_validation", "robust_score", "fold_name"], ascending=[False, False, True]), VALIDATION_COLUMNS + ["selected_by_multifold_validation", "config_id"], max_rows=120)}
"""
    experiment_md = f"""# LJN Integration Experiment

Selection is validation-only across multiple folds. Test-period metrics are recorded once after validation selection and are not used for tuning.

{_markdown_table(aggregate_out, AGGREGATE_COLUMNS + ["robust_score", "config_id"], max_rows=80)}
"""
    (output_dir / "ljn_multifold_validation.md").write_text(validation_md, encoding="utf-8")
    (output_dir / "ljn_integration_experiment.md").write_text(experiment_md, encoding="utf-8")
    _upsert_summary_section(
        summary_path,
        build_ljn_summary_section(
            aggregate_out,
            current_main_total_return=current_main_total_return,
            current_main_excess=current_main_excess,
            current_main_max_drawdown=current_main_max_drawdown,
        ),
    )
    return validation_out, aggregate_out
