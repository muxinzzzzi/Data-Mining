# Model Stability Discussion

## Purpose of the Stability Audit

The model stability audit separates two questions that are often conflated in machine-learning trading projects. The first question is whether a model obtains strong validation prediction metrics. The second question is whether the same model produces stable out-of-sample signals that improve the trading strategy. In this project, the two answers are not the same. The validation-best prediction model is not the final strategy model.

The audit uses the frozen validation and test results in `outputs/current/`. It does not retrain any model. The representative stability scores are:

| Model | Horizon | Role | Validation score | Test score | Score gap |
| --- | ---: | --- | ---: | ---: | ---: |
| MLP | 20D | Validation-best prediction model | 0.7995 | 0.4151 | 0.3845 |
| LightGBM | 5D | Main risk-budget strategy model | 0.5204 | 0.4281 | 0.0923 |
| RandomForest | 5D | Early-stress strategy model | 0.4773 | 0.5023 | -0.0250 |
| LogisticRegression | 5D | Linear baseline | 0.5636 | 0.4805 | 0.0831 |
| RidgeClassifier | 5D | Linear stability baseline | 0.5997 | 0.4752 | 0.1244 |

## MLP Validation-Test Gap

The MLP model (`mlp_small_fs30 / 20D`) is the strongest validation model, but it is also the clearest overfitting warning. Its validation selection score is 0.7995, while the matching test score falls to 0.4151. The resulting gap of 0.3845 is much larger than the gaps of the other representative models.

The target-level AUCs show the same pattern. On validation data, the MLP reports ordinary direction AUC 0.8172, clean-direction AUC 0.8977, and big-up AUC 0.8829. On test data, these fall to 0.4384, 0.5123, and 0.3467 respectively. The big-up target is especially important for this project because the decision layer is designed to preserve upside participation; the MLP's big-up test AUC below 0.50 indicates that its validation signal does not generalize reliably.

This is why the MLP is not used as the final main strategy model. Its validation strength is real inside the validation split, but it does not survive the final test period. The MLP-based upside participation strategy also underperforms buy-and-hold, with cumulative return 104.44% versus 110.87% for the benchmark and an excess return of -6.44 percentage points.

## LightGBM Stability

The final main strategy uses `LightGBM / 5D` inside `ml_risk_budget_enhancement`. LightGBM is not selected because it has the best standalone prediction AUC. In fact, its test ordinary direction AUC is 0.4734 and clean-direction AUC is 0.4434. These values are not strong enough to justify aggressive all-in/all-out timing.

The reason LightGBM is retained is strategy-layer utility. Its validation score is moderate at 0.5204 and its test score is 0.4281, giving a smaller validation-test gap of 0.0923. More importantly, once embedded in the risk-budget decision rule, it is the only real no-leverage ML strategy in the current result set that outperforms buy-and-hold. The strategy reaches cumulative return 111.35%, compared with 110.87% for buy-and-hold, for an excess return of about 0.48 percentage points.

This means LightGBM should be interpreted as an auxiliary risk-budget signal, not as a high-confidence daily directional forecaster. Its value comes from how the decision layer uses it: exposure remains close to the benchmark most of the time, while the ML signal contributes to mild recovery and confirmation logic.

## RandomForest Stability

`RandomForest / 5D` is used by the early-stress risk-budget strategy. Its validation score is 0.4773 and its test score is 0.5023, so the test score is slightly higher than validation. This makes RandomForest the most stable representative model in the score-gap view.

The strategy outcome is different from the LightGBM case. `ml_early_stress_risk_budget` does not outperform buy-and-hold on cumulative return, but it does slightly improve maximum drawdown. Its cumulative return is 110.82%, excess return is -0.06 percentage points, and maximum drawdown improves from -17.26% to -17.07%. This supports a more conservative interpretation: RandomForest is useful in the early-stress framework as a drawdown-control component, but not as the best return-maximizing strategy model.

## Prediction Accuracy Is Not Trading Utility

The key empirical lesson is that prediction metrics and trading utility are related but not equivalent. The MLP has the best validation prediction score but poor test generalization and weak realized strategy utility. Linear baselines such as LogisticRegression and RidgeClassifier show better stability on some test AUC metrics, but their risk-budget strategy variants still fail to beat buy-and-hold.

Conversely, LightGBM 5D is not the strongest standalone predictor, yet it produces the best realized strategy-layer utility when placed inside a high-participation risk-budget rule. This result is economically intuitive. In a strongly rising test period, an ML model can harm performance if it cuts exposure too often, even when some prediction metrics look attractive. A useful index-enhancement signal must avoid unnecessary missed upside as much as it avoids downside.

For the final report, the main conclusion should be stated as follows: the project does not claim that the ML model is a robust daily market-timing engine. Instead, it shows that a moderate prediction signal can be made more useful through a conservative decision layer that prioritizes benchmark participation, limits turnover, and treats ML as a risk-budget input rather than a standalone trading command.

