# Model Stability Audit

Test period: 2025-01-01 to 2026-05-06

This audit compares validation and test prediction behavior by model, horizon, and target. High validation scores with weak matching test scores are treated as overfit risk, not deployable evidence.

## Largest Validation-Test Gaps
| model_name | horizon | target_name | validation_score | test_score | score_gap | validation_auc | test_auc | auc_gap | validation_logloss | test_logloss | logloss_gap | is_validation_prediction_best | is_final_strategy_used | final_strategy_name | stability_comment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_small_fs30 | 20 | target_big_down | 0.918358 | 0.324542 | 0.593817 | 0.918358 | 0.324542 | 0.593817 | 2.018516 | 1.937450 | 0.081066 | True | False |  | validation overfit / weak test generalization |
| mlp_small_fs30 | 20 | target_big_up | 0.882913 | 0.346708 | 0.536205 | 0.882913 | 0.346708 | 0.536205 | 0.595772 | 2.455009 | -1.859236 | True | False |  | validation overfit / weak test generalization |
| LightGBMRegularized | 20 | target_big_down | 0.901147 | 0.426614 | 0.474534 | 0.901147 | 0.426614 | 0.474534 | 0.474770 | 0.662483 | -0.187713 | False | False |  | validation overfit / weak test generalization |
| XGBoostRegularized | 20 | target_big_down | 0.894969 | 0.427968 | 0.467001 | 0.894969 | 0.427968 | 0.467001 | 0.429232 | 0.683292 | -0.254060 | False | False |  | validation overfit / weak test generalization |
| GradientBoosting | 20 | target_big_down | 0.903133 | 0.437610 | 0.465524 | 0.903133 | 0.437610 | 0.465524 | 0.498674 | 0.580665 | -0.081991 | False | False |  | validation overfit / weak test generalization |
| XGBoost | 20 | target_ret | 0.395853 | -0.062241 | 0.458094 | NA | NA | NA | NA | NA | NA | False | False |  | validation overfit / weak test generalization |
| XGBoost | 20 | target_big_down | 0.894528 | 0.444940 | 0.449588 | 0.894528 | 0.444940 | 0.449588 | 0.476550 | 0.713613 | -0.237063 | False | False |  | validation overfit / weak test generalization |
| LightGBM | 20 | target_big_down | 0.906884 | 0.467530 | 0.439354 | 0.906884 | 0.467530 | 0.439354 | 0.479891 | 0.661120 | -0.181229 | False | False |  | validation overfit / weak test generalization |
| mlp_small_fs30 | 20 | target_ret | 0.394366 | -0.025811 | 0.420177 | NA | NA | NA | NA | NA | NA | True | False |  | validation overfit / weak test generalization |
| HistGradientBoosting | 20 | target_big_down | 0.881289 | 0.462072 | 0.419217 | 0.881289 | 0.462072 | 0.419217 | 0.557828 | 0.696089 | -0.138261 | False | False |  | validation overfit / weak test generalization |
| ExtraTreesDeep | 20 | target_big_down | 0.893645 | 0.476096 | 0.417550 | 0.893645 | 0.476096 | 0.417550 | 0.634843 | 0.668636 | -0.033793 | False | False |  | validation overfit / weak test generalization |
| mlp_tiny | 20 | target_big_down | 0.825684 | 0.412669 | 0.413015 | 0.825684 | 0.412669 | 0.413015 | 1.468601 | 1.054932 | 0.413669 | False | False |  | validation overfit / weak test generalization |
| RandomForestDeep | 20 | target_big_down | 0.889232 | 0.486693 | 0.402539 | 0.889232 | 0.486693 | 0.402539 | 0.594136 | 0.715586 | -0.121450 | False | False |  | validation overfit / weak test generalization |
| GradientBoosting | 20 | target_trade | 0.840794 | 0.440723 | 0.400071 | 0.840794 | 0.440723 | 0.400071 | 0.588961 | 0.674478 | -0.085517 | False | False |  | validation overfit / weak test generalization |
| ExtraTrees | 20 | target_big_down | 0.849515 | 0.457291 | 0.392224 | 0.849515 | 0.457291 | 0.392224 | 0.910294 | 0.764682 | 0.145612 | False | False |  | validation overfit / weak test generalization |
| RandomForestDeep | 20 | target_trade | 0.827937 | 0.435946 | 0.391990 | 0.827937 | 0.435946 | 0.391990 | 0.586687 | 0.793138 | -0.206451 | False | False |  | validation overfit / weak test generalization |
| XGBoostRegularized | 20 | target_trade | 0.846349 | 0.456522 | 0.389827 | 0.846349 | 0.456522 | 0.389827 | 0.632553 | 0.832376 | -0.199823 | False | False |  | validation overfit / weak test generalization |
| LightGBM | 20 | target_ret | 0.384754 | -0.004705 | 0.389459 | NA | NA | NA | NA | NA | NA | False | False |  | validation overfit / weak test generalization |
| mlp_small_fs30 | 20 | target_clean_direction | 0.897727 | 0.512283 | 0.385444 | 0.897727 | 0.512283 | 0.385444 | 0.864457 | 2.398412 | -1.533955 | True | False |  | validation overfit / weak test generalization |
| LightGBMRegularized | 20 | target_up | 0.836627 | 0.455033 | 0.381595 | 0.836627 | 0.455033 | 0.381595 | 0.604565 | 0.802410 | -0.197845 | False | False |  | validation overfit / weak test generalization |
| mlp_small_fs30 | 20 | target_up | 0.817194 | 0.438385 | 0.378809 | 0.817194 | 0.438385 | 0.378809 | 0.819391 | 1.890013 | -1.070622 | True | False |  | validation overfit / weak test generalization |
| mlp_tiny | 20 | target_ret | 0.171160 | -0.206459 | 0.377619 | NA | NA | NA | NA | NA | NA | False | False |  | weak or unstable predictive evidence |
| XGBoostRegularized | 20 | target_up | 0.838274 | 0.463452 | 0.374822 | 0.838274 | 0.463452 | 0.374822 | 0.593281 | 0.796563 | -0.203282 | False | False |  | validation overfit / weak test generalization |
| RandomForest | 20 | target_big_down | 0.858782 | 0.492351 | 0.366431 | 0.858782 | 0.492351 | 0.366431 | 0.780521 | 0.712343 | 0.068177 | False | False |  | validation overfit / weak test generalization |
| RandomForest | 20 | target_trade | 0.842222 | 0.476607 | 0.365615 | 0.842222 | 0.476607 | 0.365615 | 0.653544 | 0.693135 | -0.039591 | False | False |  | validation overfit / weak test generalization |
| XGBoost | 20 | target_up | 0.835968 | 0.473275 | 0.362694 | 0.835968 | 0.473275 | 0.362694 | 0.605068 | 0.791160 | -0.186093 | False | False |  | validation overfit / weak test generalization |
| HistGradientBoosting | 20 | target_ret | 0.353770 | -0.003748 | 0.357518 | NA | NA | NA | NA | NA | NA | False | False |  | validation overfit / weak test generalization |
| LightGBM | 20 | target_up | 0.821805 | 0.464441 | 0.357364 | 0.821805 | 0.464441 | 0.357364 | 0.623661 | 0.815290 | -0.191629 | False | False |  | validation overfit / weak test generalization |
| RandomForestDeep | 20 | target_up | 0.813900 | 0.461602 | 0.352298 | 0.813900 | 0.461602 | 0.352298 | 0.607041 | 0.762247 | -0.155206 | False | False |  | validation overfit / weak test generalization |
| XGBoostRegularized | 20 | target_ret | 0.366944 | 0.015032 | 0.351911 | NA | NA | NA | NA | NA | NA | False | False |  | validation overfit / weak test generalization |
| HistGradientBoosting | 20 | target_up | 0.755270 | 0.409427 | 0.345843 | 0.755270 | 0.409427 | 0.345843 | 0.669477 | 0.851868 | -0.182391 | False | False |  | validation overfit / weak test generalization |
| RandomForest | 20 | target_up | 0.778656 | 0.434430 | 0.344226 | 0.778656 | 0.434430 | 0.344226 | 0.649555 | 0.706821 | -0.057266 | False | False |  | validation overfit / weak test generalization |
| XGBoost | 20 | target_trade | 0.805714 | 0.467912 | 0.337802 | 0.805714 | 0.467912 | 0.337802 | 0.709366 | 0.852514 | -0.143148 | False | False |  | validation overfit / weak test generalization |
| LightGBMRegularized | 20 | target_trade | 0.809841 | 0.473484 | 0.336357 | 0.809841 | 0.473484 | 0.336357 | 0.688794 | 0.814166 | -0.125372 | False | False |  | validation overfit / weak test generalization |
| mlp_small_fs30 | 20 | target_trade | 0.835238 | 0.499265 | 0.335973 | 0.835238 | 0.499265 | 0.335973 | 1.252516 | 1.960525 | -0.708009 | True | False |  | validation overfit / weak test generalization |
| LightGBM | 20 | target_trade | 0.802222 | 0.469810 | 0.332412 | 0.802222 | 0.469810 | 0.332412 | 0.707214 | 0.807153 | -0.099939 | False | False |  | validation overfit / weak test generalization |
| mlp_small | 20 | target_clean_direction | 0.779221 | 0.449066 | 0.330154 | 0.779221 | 0.449066 | 0.330154 | 2.576129 | 3.828965 | -1.252836 | False | False |  | validation overfit / weak test generalization |
| ExtraTreesDeep | 10 | target_trade | 0.739956 | 0.415459 | 0.324497 | 0.739956 | 0.415459 | 0.324497 | 0.643534 | 0.731733 | -0.088198 | False | False |  | validation overfit / weak test generalization |
| ExtraTreesDeep | 20 | target_trade | 0.778095 | 0.458849 | 0.319246 | 0.778095 | 0.458849 | 0.319246 | 0.670646 | 0.735846 | -0.065199 | False | False |  | validation overfit / weak test generalization |
| GradientBoosting | 20 | target_up | 0.740777 | 0.425054 | 0.315723 | 0.740777 | 0.425054 | 0.315723 | 0.596297 | 0.651540 | -0.055242 | False | False |  | validation overfit / weak test generalization |
