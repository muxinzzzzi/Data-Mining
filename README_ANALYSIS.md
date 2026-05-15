# Enhanced Index Workflow

This project now uses a prediction-first machine-learning pipeline followed by a validation-tuned strategy layer.

## How To Run

```bash
python run_enhanced_strategy.py
```

If the local shell does not provide a `python` command, use:

```bash
python3 run_enhanced_strategy.py
```

The full runner uses existing prediction artifacts when they are already present. To force a full prediction rebuild before strategy tuning:

```bash
FORCE_REBUILD_PREDICTIONS=1 python3 run_enhanced_strategy.py
```

To run underperformance diagnostics only, without rebuilding predictions or retuning strategy parameters:

```bash
python3 run_enhanced_strategy.py --diagnose-only
```

## Current Structure

- `src/`: leakage-safe data loading, feature engineering, targets, models, preprocessing, feature selection, walk-forward prediction, and metrics.
- `src/strategy/`: decision rules, validation tuning, backtest, strategy metrics, attribution, plots, and report writing.
- `scripts/run_prediction_only.py`: prediction-only workflow.
- `scripts/run_full_workflow.py`: prediction artifacts plus strategy tuning/backtest/report workflow.
- `run_enhanced_strategy.py`: root entrypoint for the current full workflow.
- `archive/old_strategy_code/`: old strategy/report files and historical strategy outputs.

## Output Highlights

- Prediction artifacts: `outputs/predictions/`, `outputs/metrics/`, `outputs/diagnostics/`
- Strategy files: `outputs/strategy_daily_all.csv`, `outputs/strategy_metrics_all.csv`, `outputs/strategy_tuned_params.json`
- Attribution: `outputs/main_strategy_attribution.csv`, `outputs/main_strategy_attribution_summary.md`
- Reports: `outputs/summary.md`, `outputs/final_report_material.md`
- Plots: `outputs/plots/`

## Modeling Note

The current out-of-sample signal is moderate. Ordinary direction AUC and clean-direction AUC are below 0.60, while big-up AUC is above 0.60. The strategy layer therefore uses benchmark-aware index enhancement and strong-upside participation signals rather than aggressive all-in/all-out market timing.
