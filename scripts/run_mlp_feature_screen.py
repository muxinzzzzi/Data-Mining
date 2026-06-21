from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import METRICS_DIR, MIN_TRAIN_SAMPLES, PREDICTION_DIR, VALID_END, VALID_RETRAIN_EVERY, VALID_START
from src.data_loader import load_data
from src.features import build_daily_features, build_intraday_features, get_feature_columns
from src.metrics import compute_model_metrics
from src.models import build_mlp_feature_screen_specs
from src.utils import write_json_ready
from src.walk_forward import walk_forward_predictions


BASELINE_MODELS = ["GradientBoosting", "mlp_small", "mlp_tiny"]
SCREEN_COLUMNS = [
    "model_name",
    "horizon",
    "selection_score",
    "AUC",
    "clean_direction_AUC",
    "trade_AUC",
    "big_up_AUC",
    "big_down_AUC",
    "return_correlation",
]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(write_json_ready(value), indent=2, ensure_ascii=False), encoding="utf-8")


def _selected_outer_top_k() -> int | None:
    path = METRICS_DIR / "feature_top_k_selection.json"
    if not path.exists():
        raise FileNotFoundError("Run scripts/run_prediction_only.py before running the MLP feature screen.")
    return json.loads(path.read_text(encoding="utf-8"))["selected_top_k"]


def _fmt(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def _best_row(metrics: pd.DataFrame, model_filter: pd.Series) -> dict[str, Any]:
    rows = metrics.loc[model_filter].sort_values("selection_score", ascending=False)
    return rows.iloc[0].to_dict() if not rows.empty else {}


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    outer_top_k = _selected_outer_top_k()

    validation_metrics_path = METRICS_DIR / "validation_model_metrics_by_horizon.csv"
    if not validation_metrics_path.exists():
        raise FileNotFoundError("No official validation metrics found; run scripts/run_prediction_only.py first.")

    print("Loading features for validation-only MLP feature screen...")
    daily, m5 = load_data()
    feature_df = build_daily_features(daily, build_intraday_features(m5))
    feature_cols = get_feature_columns(feature_df)
    specs = build_mlp_feature_screen_specs(outer_top_k)
    print(f"Outer selected feature_top_k: {outer_top_k}")
    print(f"MLP internal feature candidates: {[spec.name for spec in specs]}")

    predictions = walk_forward_predictions(
        feature_df,
        feature_cols,
        specs,
        VALID_START,
        VALID_END,
        VALID_RETRAIN_EVERY,
        min_train_samples=MIN_TRAIN_SAMPLES,
        label="validation_mlp_feature_screen",
        feature_top_k=outer_top_k,
    )
    screened_metrics = compute_model_metrics(predictions)
    baseline_metrics = pd.read_csv(validation_metrics_path)
    baseline_metrics = baseline_metrics.loc[baseline_metrics["model_name"].isin(BASELINE_MODELS)].copy()
    combined = pd.concat([baseline_metrics, screened_metrics], ignore_index=True).sort_values(
        ["selection_score", "clean_direction_AUC", "AUC"],
        ascending=[False, False, False],
        na_position="last",
    )

    predictions.to_csv(PREDICTION_DIR / "validation_predictions_mlp_feature_screen.csv", index=False)
    combined.to_csv(METRICS_DIR / "mlp_feature_set_validation_screen.csv", index=False)

    best_candidate = _best_row(combined, combined["model_name"].str.startswith("mlp_small_fs"))
    best_current_mlp = _best_row(combined, combined["model_name"].isin(["mlp_small", "mlp_tiny"]))
    gradient_boosting = _best_row(combined, combined["model_name"] == "GradientBoosting")
    export = {
        "selection_basis": "validation_only_mlp_feature_screen",
        "note": "The test period is not used in this screen. Each MLP SelectKBest selector and StandardScaler are fitted inside each validation walk-forward training window.",
        "outer_feature_top_k": outer_top_k,
        "internal_feature_candidates": [int(spec.name.rsplit("fs", 1)[1]) for spec in specs],
        "screened_models": [spec.name for spec in specs],
        "best_feature_screen_candidate": best_candidate,
        "best_current_mlp_baseline": best_current_mlp,
        "gradient_boosting_reference": gradient_boosting,
    }
    _write_json(METRICS_DIR / "mlp_feature_set_validation_screen.json", export)

    lines = [
        "# MLP Feature Set Validation Screen",
        "",
        "- Selection basis: validation period only.",
        f"- Outer feature_top_k retained from current pipeline: {outer_top_k}.",
        f"- MLP-only inner selectors: {', '.join(spec.name.rsplit('fs', 1)[1] for spec in specs)} features; each fitted within its walk-forward training window.",
        "- No test-period metrics are used to choose a feature set in this screen.",
        "",
        "| model | horizon | score | AUC | clean AUC | trade AUC | return corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    best_per_model = (
        combined.sort_values("selection_score", ascending=False)
        .drop_duplicates("model_name")
        .loc[:, SCREEN_COLUMNS]
        .sort_values("selection_score", ascending=False)
    )
    for row in best_per_model.itertuples(index=False):
        lines.append(
            f"| {row.model_name} | {int(row.horizon)} | {_fmt(row.selection_score)} | {_fmt(row.AUC)} | "
            f"{_fmt(row.clean_direction_AUC)} | {_fmt(row.trade_AUC)} | {_fmt(row.return_correlation)} |"
        )
    (METRICS_DIR / "mlp_feature_set_validation_screen.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n===== MLP FEATURE SET VALIDATION SCREEN =====")
    print(f"Best feature-screen candidate: {best_candidate.get('model_name')} / {best_candidate.get('horizon')}D / score={_fmt(best_candidate.get('selection_score'))}")
    print(f"Best current MLP baseline: {best_current_mlp.get('model_name')} / {best_current_mlp.get('horizon')}D / score={_fmt(best_current_mlp.get('selection_score'))}")
    print(f"GradientBoosting reference: {gradient_boosting.get('model_name')} / {gradient_boosting.get('horizon')}D / score={_fmt(gradient_boosting.get('selection_score'))}")
    print(f"Outputs saved to: {METRICS_DIR}")
    print("=============================================")


if __name__ == "__main__":
    main()
