# Prediction Summary

## Run Scope
- Feature count: 267
- Selected feature_top_k: 120
- Model families used: HistGradientBoosting, RandomForest, ExtraTrees, GradientBoosting, ExtraTreesDeep, RandomForestDeep
- Horizons: 1, 3, 5, 10, 20

## Best Validation Selection
- Best validation model: GradientBoosting
- Best validation horizon: 20
- Best validation ordinary AUC: 0.8276
- Best validation clean_direction_AUC: 0.7837
- Validation selection score: 0.7293

## Test Prediction Metrics
- Best test ordinary AUC: 0.5562 (HistGradientBoosting, 1D)
- Best test clean_direction_AUC: 0.5742 (RandomForest, 3D)
- Best test trade_AUC: 0.5612 (HistGradientBoosting, 3D)
- Best test big_up_AUC: 0.6163 (HistGradientBoosting, 3D)
- Best test big_down_AUC: 0.5716 (RandomForest, 1D)
- Best test return_correlation: 0.1626 (RandomForest, 20D)
- Clean-direction AUC >= 0.60: No

## Leakage-Control Checklist
- No random train/test split is used.
- Validation period and test period are fixed by configuration.
- Feature_top_k is selected using validation predictions only.
- Model preference is selected using validation prediction metrics only.
- Each walk-forward chunk trains only on rows with target_end_date < chunk_start.
- Feature selection is refit inside each walk-forward chunk using only that chunk's training rows.
- Quantile clipping and median imputation are fitted only on the relevant task's training rows.
- Historical target quantiles only use returns whose target_end_date is before the current date.
- Test-set information is not used for feature selection, model selection, preprocessing, or threshold construction.

## Interpretation
- Raw up/down AUC may remain low because near-zero returns are noisy.
- clean_direction_AUC is the denoised directional metric because middle noisy samples are ignored.
- trade_AUC and tail-target AUC are more relevant for later strategy design than raw up/down alone.
