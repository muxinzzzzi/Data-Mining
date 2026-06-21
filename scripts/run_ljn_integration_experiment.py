from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUT_DIR
from src.data_loader import load_data
from src.features import build_daily_features, build_intraday_features, get_feature_columns
from src.strategy.decision_rules import RiskBudgetParams
from src.strategy.ljn_integration import run_ljn_integration_experiment
from src.strategy.ljn_integration_reports import write_ljn_integration_reports


CURRENT_DIR = OUTPUT_DIR / "current"
CURRENT_MISC_DIR = CURRENT_DIR / "misc"
CURRENT_PARAMS_DIR = CURRENT_DIR / "params"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected_feature_top_k(path: Path) -> int | None:
    if not path.exists():
        return None
    data = _read_json(path)
    value = data.get("selected_top_k")
    return None if value is None else int(value)


def _load_risk_budget_params(path: Path) -> RiskBudgetParams:
    data = _read_json(path)
    params = data.get("selected_params") or data.get("selected", {}).get("params")
    if not params:
        raise RuntimeError(f"No selected risk-budget params found in {path}")
    return RiskBudgetParams(**params)


def _load_current_main_metrics(path: Path) -> tuple[float, float, float]:
    metrics = pd.read_csv(path)
    row = metrics.loc[metrics["strategy"] == "ml_risk_budget_enhancement"]
    if row.empty:
        raise RuntimeError(f"`ml_risk_budget_enhancement` not found in {path}")
    main = row.iloc[0]
    return (
        float(main["total_return"]),
        float(main["excess_return_vs_buy_hold"]),
        float(main["max_drawdown"]),
    )


def main() -> None:
    print("Loading current feature frame and prediction artifacts...")
    daily, m5 = load_data()
    feature_df = build_daily_features(daily, build_intraday_features(m5))
    feature_cols = get_feature_columns(feature_df)

    validation_predictions = pd.read_csv(CURRENT_MISC_DIR / "validation_predictions_all_models.csv")
    test_predictions = pd.read_csv(CURRENT_MISC_DIR / "test_predictions_all_models.csv")
    risk_budget_params = _load_risk_budget_params(CURRENT_PARAMS_DIR / "risk_budget_tuned_params.json")
    feature_top_k = _selected_feature_top_k(CURRENT_PARAMS_DIR / "feature_top_k_selection.json")
    current_main_total_return, current_main_excess, current_main_max_drawdown = _load_current_main_metrics(
        CURRENT_DIR / "strategy_metrics_all.csv"
    )

    print("Running controlled ljn-inspired multi-fold experiment...")
    validation, aggregate = run_ljn_integration_experiment(
        feature_df=feature_df,
        feature_cols=feature_cols,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        risk_budget_params=risk_budget_params,
        feature_top_k=feature_top_k,
        current_main_total_return=current_main_total_return,
        cache_dir=CURRENT_MISC_DIR,
    )

    print("Writing ljn integration reports under outputs/current/...")
    write_ljn_integration_reports(
        validation=validation,
        aggregate=aggregate,
        output_dir=CURRENT_DIR,
        summary_path=CURRENT_DIR / "summary.md",
        current_main_total_return=current_main_total_return,
        current_main_excess=current_main_excess,
        current_main_max_drawdown=current_main_max_drawdown,
    )
    selected = aggregate.loc[aggregate["selected_by_multifold_validation"].astype(bool)].iloc[0]
    print(
        "Selected by multi-fold validation: "
        f"{selected['strategy_name']} / {selected['model_name']} / {int(selected['horizon'])}D, "
        f"test_total_return={selected['test_total_return']:.6f}, "
        f"test_excess={selected['test_excess_return_vs_buy_hold']:.6f}"
    )
    print(f"Wrote: {CURRENT_DIR / 'ljn_multifold_validation.csv'}")
    print(f"Wrote: {CURRENT_DIR / 'ljn_integration_experiment.csv'}")


if __name__ == "__main__":
    main()
