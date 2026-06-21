# LightGBM 5D Feature Importance Missing

Figure 7 (`fig7_lightgbm_feature_importance.png`) was not generated because the current `outputs/current/` artifacts do not contain a saved LightGBM 5D model object or a saved feature-importance table for the final `ml_risk_budget_enhancement` strategy.

To preserve the no-retraining rule for the final report material pass, this script did not refit LightGBM or recompute feature importance.

Recommended report handling: mention that feature importance was unavailable in the frozen output bundle and keep the model-stability and strategy-attribution figures as the interpretability evidence.
