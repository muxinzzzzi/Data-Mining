# Enhanced Index Strategy Summary

## Setup
- Initial capital: RMB 100,000
- Test period: 2025-01-01 to 2026-05-06
- Transaction cost rate: 0.0010
- Signal timing: predictions and regime filters are observed after close on date t and applied to the next tradable return.

## Prediction Interpretation
- Best test ordinary direction AUC: 0.5797
- Best test clean-direction AUC: 0.5757
- Best test big-up AUC: 0.6384
- Ordinary direction AUC and clean-direction AUC are below 0.60, so the model is not used as an aggressive 0/1 market-timing engine.
- Big-up AUC is above 0.60, so the decision layer focuses on strong-upside participation and risk-aware exposure adjustment.

## Prediction Model Selection
- MLP candidates evaluated: mlp_small, mlp_small_fs30, mlp_tiny
- Validation-selected model/horizon: mlp_small_fs30 / 20D
- MLP selected by validation score: Yes
- Selected-model validation score: 0.7995
- Selected-model matching test score: 0.4151
- Selected-model test direction AUC: 0.4384
- Selected-model test clean-direction AUC: 0.5123
- Selected-model test trade AUC: 0.4993
- Selected-model test return correlation: -0.0258
- Best MLP comparison: mlp_small_fs30 / 20D; validation score 0.7995; matching test score 0.4151; test AUC 0.4384; test clean-direction AUC 0.5123; test trade AUC 0.4993; test return correlation -0.0258


## Strategy Results
- Buy-and-hold total return: 1.1087
- Buy-and-hold max drawdown: -0.1726
- Best no-leverage strategy: ml_risk_budget_enhancement
- Whether best no-leverage strategy is benchmark clone: False
- Best real no-leverage ML strategy: ml_risk_budget_enhancement
- ML strategy total return: 1.1135
- ML strategy excess return: 0.0048
- ML strategy max drawdown: -0.1726
- Best enhanced-exposure strategy: ml_big_up_plus_115
- Best enhanced-exposure excess return: -0.1342
- Any real no-leverage ML strategy outperforms buy-and-hold: Yes

## Interpretation
No-leverage and enhanced-exposure strategies are reported separately. If no-leverage ML enhancement fails to outperform buy-and-hold, the main reason is that the test period was strongly upward and even mild defensive exposure cuts can miss upside. If an enhanced-exposure strategy outperforms, that result should be interpreted separately because it allows exposure above 1.00.

## Tradeable-Up Signal Validation
- Selected signal: pred_big_up_prob
- Selected model/horizon: mlp_small_fs30, 20D
- Validation top-bottom spread: 0.0135
- Validation top group win rate: 0.6800
- Test top-bottom spread: -0.0024
- Test top group win rate: 0.5469
- Low-score group really corresponds to future decline: No
- Conclusion: The strategy should not rely on low-score forced selling.


## Upside Participation Index Enhancement
- Strategy: ml_upside_participation_enhancement
- Design: benchmark-aware no-leverage enhancement, not aggressive 0/1 buy/sell timing.
- Participation rule: default high participation; high upside signal keeps exposure at 1.00.
- Defensive rule: low signal does not directly sell; only low signal plus weak trend and high risk can mildly reduce exposure.
- Maximum exposure: 1.00, no leverage.
- Parameter selection: validation-period only.
- Total return: 1.0444
- Excess return: -0.0644
- Max drawdown: -0.1726
- Sharpe: 2.2731
- Average position: 0.9872
- Days below full exposure: 41
- Total turnover: 1.8000
- Missed upside: 0.0508
- Avoided downside: 0.0198
- Transaction cost: 0.0018
- Mainly missed upside: Yes, missed upside is larger than avoided downside plus transaction cost.
- Mainly transaction cost: No
- Outperforms buy-and-hold: No
- Becomes best real no-leverage ML timing strategy: No
- Strong rising test period makes defensive exposure cuts naturally costly: Yes

The upside participation enhancement strategy improves the interpretability of ML-based index enhancement and maintains high benchmark participation, but it still does not outperform buy-and-hold in this strongly rising test period.


## Stability-Aware High-Participation Index Enhancement
- Strategy: ml_stability_aware_high_participation
- Design: not direct buy/sell and not low-score forced selling.
- Interpretation: high-participation benchmark-aware no-leverage enhancement.
- Default exposure: 1.00.
- Defensive rule: only when ensemble upside score is very low and trend or risk confirms weakness does exposure get mildly reduced.
- Ensemble purpose: multiple validation-selected upside signals reduce single-validation winner-take-all overfitting.
- Signal selection: validation-period only.
- Selected signal components: mlp_small_fs30 20D pred_big_up_prob score=0.6600; mlp_small_fs30 20D pred_up_prob score=0.6393; mlp_small_fs30 20D pred_trade_prob score=0.5525
- Total return: 1.1087
- Excess return: 0.0000
- Max drawdown: -0.1726
- Sharpe: 2.3256
- Average position: 1.0000
- Days below full exposure: 0
- Total turnover: 0.0000
- Missed upside: 0.0000
- Avoided downside: 0.0000
- Transaction cost: 0.0000
- Outperforms buy-and-hold: No
- Becomes best real no-leverage ML timing strategy: No

The strategy maintains high benchmark participation and reduces dependence on low-score sell signals, but it still does not outperform buy-and-hold in this strongly rising test period. The attribution indicates that missed upside remains larger than avoided downside.


## Risk-Budget Drawdown Control Enhancement
- Strategy: ml_risk_budget_enhancement
- Design: risk-budget index enhancement, not 0/1 buy/sell timing.
- Participation rule: default high index participation near 1.00.
- Defensive rule: exposure is lowered only when volatility, drawdown, trend, and tail-risk states jointly deteriorate.
- ML role: auxiliary recovery or confirmation signal only; low ML probability alone does not force selling.
- Maximum exposure: 1.00, no leverage and no shorting.
- Parameter selection: validation-period only.
- Selected model/horizon: LightGBM / 5D
- Total return: 1.1135
- Excess return: 0.0048
- Max drawdown: -0.1726
- Drawdown improvement vs buy-and-hold: 0.0000
- Sharpe: 2.4234
- Average position: 0.9764
- Days below full exposure: 46
- Total turnover: 5.0000
- Missed upside: 0.0659
- Avoided downside: 0.0694
- Transaction cost: 0.0050
- Outperforms buy-and-hold: Yes
- Reduces maximum drawdown: No
- Becomes best real no-leverage ML timing strategy: Yes

The risk-budget enhancement strategy provides an explicit drawdown-control framework, but the realized maximum drawdown remains close to buy-and-hold because the test period is strongly upward-trending and risk-off periods are followed by rapid rebounds.


## Early-Stress Risk Budget Enhancement
- Strategy: ml_early_stress_risk_budget
- Design: high-participation index enhancement, not 0/1 buy/sell timing.
- Early stress feature categories: short_term_downside_acceleration: 11; intraday_selling_pressure: 7; short_term_liquidity_volume_pressure: 8; early_stress_composite: 5
- Early stress feature count: 31
- Stress logic: early_stress_score attempts to detect short-term downside pressure, intraday selling stress, liquidity pressure, weak trend, and volatility expansion.
- Defensive rule: reduce exposure only when early stress is high and ML/upside signals are not strong.
- Maximum exposure: 1.00, no leverage and no shorting.
- Parameter selection: validation-period only.
- Selected model/horizon: RandomForest / 5D
- Total return: 1.1082
- Excess return: -0.0006
- Max drawdown: -0.1707
- Drawdown improvement vs buy-and-hold: 0.0019
- Sharpe: 2.3497
- Average position: 0.9885
- Days below full exposure: 26
- Total turnover: 3.1200
- Outperforms buy-and-hold: No
- Reduces maximum drawdown: Yes
- Becomes best real no-leverage ML timing strategy: No
- Reduced exposure before benchmark trough: Yes
- Earlier than original risk budget in max-drawdown event: Yes

The early-stress risk-budget strategy improves the original risk-budget design by reacting earlier to short-term downside pressure and intraday selling stress.
## External Market Data Enhancement
- External data used: index_daily, margin_daily, shibor_daily; all shifted by one raw market day before merge.
- Ablation variants: hard external index_only/index_margin/index_shibor/all, early_stress index_only/all, and soft-confirm/soft-lite all/index_only.
- Best external strategy by total return: `ml_external_soft_lite_risk_budget_index_only`.
- Best external total return: 1.1061; excess vs buy-and-hold: -0.0026.
- Best external max drawdown strategy: `ml_external_early_stress_risk_budget_all`; max drawdown: -0.1480.
- Best external Sharpe strategy: `ml_external_risk_budget_enhancement_index_margin`; Sharpe: 2.4256.
- External best-return strategy beats the best original risk strategy: No.
- External variants reduce max drawdown vs buy-and-hold: Yes.
- Tradeoff to inspect: external filters lower exposure when market-wide risk score rises, so they can reduce drawdown or volatility at the cost of missed upside in rebound periods.

## External Soft Confirmation Strategy
- Motivation: the hard external filter reduced drawdown but cut exposure on too many rebound/upside days, creating excessive missed upside.
- Soft design: external risk is only a confirmation layer. `watch` does not cut exposure; `high/extreme` risk receives only mild weights and needs internal stress confirmation for larger cuts.
- Best original soft-confirm strategy by total return: `ml_external_soft_confirm_early_stress_index_only`.
- Best original soft-confirm total return: 1.0938; excess vs buy-and-hold: -0.0150.
- Best soft strategy overall by total return: `ml_external_soft_lite_risk_budget_index_only`.
- Best soft total return overall: 1.1061; excess vs buy-and-hold: -0.0026.
- Best soft max drawdown strategy overall: `ml_external_soft_confirm_early_stress_index_only`; max drawdown: -0.1707.
- Best soft Sharpe strategy overall: `ml_external_soft_lite_risk_budget_index_only`; Sharpe: 2.4192; Calmar: 4.9456.
- Soft best-return strategy beats buy-and-hold: No.
- Soft variants reduce max drawdown vs buy-and-hold: Yes.
- Soft best-return total-return delta vs original risk-budget: -0.0074.
- Soft best-drawdown improvement vs hard best-drawdown strategy: -0.0227.
- Missed-upside delta vs hard best-return strategy: -0.1211; avoided-downside delta: -0.0652.
- Recommended main result: keep the original `ml_risk_budget_enhancement` if total return is the priority; report the best soft confirmation strategy as a supplementary drawdown-control experiment if it improves drawdown with materially less missed upside than the hard filter.

## External Soft-Lite Confirmation Strategy
- Soft-lite design: use the same external/internal confirmation logic, but with milder external weights and a tighter external overlay no-trade band so small risk discounts can be tested without hard de-risking.
- Best soft-lite strategy by total return: `ml_external_soft_lite_risk_budget_index_only`.
- Best soft-lite total return: 1.1061; excess vs buy-and-hold: -0.0026.
- Best soft-lite max drawdown strategy: `ml_external_soft_lite_early_stress_index_only`; max drawdown: -0.1707.
- Best soft-lite Sharpe strategy: `ml_external_soft_lite_risk_budget_index_only`; Sharpe: 2.4192; Calmar: 4.9456.
- Soft-lite best-return strategy beats buy-and-hold: No.
- Soft-lite best-return total-return delta vs original soft-confirm best-return: 0.0123.
## Model Stability Audit
- Validation-best prediction model: `mlp_small_fs30` / 20D.
- Validation-best prediction score: 0.7995; matching test score: 0.4151.
- MLP test generalization: direction AUC 0.4384, clean-direction AUC 0.5123, trade AUC 0.4993, return correlation -0.0258.
- Final main strategy depends on MLP: No.
- Final main strategy model/signal: `LightGBM` / 5D inside `ml_risk_budget_enhancement`.
- Prediction-layer best and strategy-layer best are the same: No.
- Interpretation: prediction-layer accuracy does not necessarily translate into strategy-layer utility.
- MLP recommendation: do not use MLP as the final main model under current evidence; treat it as a validation-overfit warning and a research candidate.
- LightGBM / 5D recommendation: keep as current deployable strategy model because it is selected by strategy-layer utility and remains the only no-leverage strategy with positive excess return.
- Best simple linear baseline by strategy utility: `RidgeClassifier` with total return 1.0720 and excess -0.0367.
- Linear baseline conclusion: simple models are useful as stability controls; they help identify whether complex validation winners add real trading value rather than only validation-period fit.
## LJN Integration Experiment
- Direct ljn merge: No. The ljn branch was used only as a candidate idea library.
- Ideas absorbed: multi-fold validation, `max_position_change` position smoothing, and reduce-position-worth auxiliary target.
- Current main strategy remains: `ml_risk_budget_enhancement` / LightGBM / 5D; test total return 1.1135, excess 0.0048, max drawdown -0.1726.
- Multi-fold selection rule: validation folds only, using `mean_validation_excess + 0.5 * min_validation_excess + 0.2 * mean_validation_calmar - 0.1 * std_validation_excess - 0.05 * mean_turnover`.
- Validation-selected candidate: `ml_risk_budget_with_position_smoothing` / `LightGBM` / 5D; mean validation excess 0.0050, min validation excess -0.0009.
- Validation stability: No.
- Test result for validation-selected candidate: total return 1.0882, excess -0.0206, max drawdown -0.1726.
- Any ljn-inspired candidate beats buy-and-hold on test: Yes.
- Any ljn-inspired candidate beats current main strategy on test: No; best test candidate by total return is `ml_risk_budget_with_position_smoothing` config `smoothing_none` at 1.1135.
- Upgrade main strategy: No; keep `ml_risk_budget_enhancement` as current main strategy.
- Reason: The selected candidate does not have positive excess in every validation fold.
- Ideas worth retaining if not upgraded: multi-fold validation as a stability gate; position smoothing as a turnover/cost control; reduce-position-worth labels as a conservative auxiliary risk signal.
