# External Market Feature Report

## Raw Data Scope
| file | exists | rows | columns | start | end |
| --- | --- | --- | --- | --- | --- |
| index_daily.csv | True | 1789 | 37 | 2019-01-02 | 2026-05-22 |
| margin_daily.csv | True | 1788 | 4 | 2019-01-02 | 2026-05-21 |
| shibor_daily.csv | True | 1840 | 5 | 2019-01-02 | 2026-05-22 |
| north_money.csv | True | 1717 | 13 | 2019-01-02 | 2026-05-22 |
| industry_daily.csv | True | 0 | 1 | NA | NA |

## Usage Decision
- Used in strategy features: `index_daily.csv`, `margin_daily.csv`, `shibor_daily.csv`.
- Excluded from the main strategy feature set: `north_money.csv` and `industry_daily.csv`.
- Reason: north-money data is not used as the main strategy feature under the current project constraint; `industry_daily.csv` is empty.
- Leakage control: every external feature is computed on the raw market timeline, shifted by one raw market day, then backward-aligned to the 880823 trading dates.

## Feature Construction
- External feature count in `feature_df`: 133.
- Index features: multi-index returns, moving-average gaps, annualized volatility, drawdown, amount change/z-score, and style relative strength.
- Margin features: balance/buy change, rolling z-score, moving-average gap, and `margin_risk_off`.
- Shibor features: rate changes, z-scores, term slopes, and `shibor_tightening`.
- Composite risk score: 7 binary conditions covering index trend, index drawdown, index volatility, index liquidity, style weakness, margin risk, and Shibor tightening.

## Test-Period Risk Score Summary
| feature | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| external_index_risk_score | 1.348910 | 1.471524 | 0.000000 | 0.000000 | 1.000000 | 2.000000 | 5.000000 |
| external_margin_risk_score | 0.183801 | 0.387926 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| external_shibor_risk_score | 0.102804 | 0.304177 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| external_market_risk_score | 1.635514 | 1.726951 | 0.000000 | 0.000000 | 1.000000 | 3.000000 | 6.000000 |

## Test-Period Risk Condition Frequency
| condition | test_period_frequency |
| --- | --- |
| external_condition_index_trend_weak | 0.433022 |
| external_condition_index_drawdown | 0.165109 |
| external_condition_index_volatility | 0.239875 |
| external_condition_index_liquidity | 0.292835 |
| external_condition_style_weak | 0.218069 |
| external_condition_margin_risk | 0.183801 |
| external_condition_shibor_tightening | 0.102804 |

## Ablation Result Summary
| strategy | total_return | excess_return_vs_buy_hold | max_drawdown | drawdown_improvement_vs_buy_hold | sharpe | calmar | average_position | missed_upside | avoided_downside | total_transaction_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| buy_hold | 1.108741 | 0.000000 | -0.172603 | 0.000000 | 2.325563 | 5.042679 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| ml_risk_budget_enhancement | 1.113495 | 0.004754 | -0.172603 | 0.000000 | 2.423439 | 4.978558 | 0.976406 | 0.065942 | 0.069375 | 0.005000 |
| ml_early_stress_risk_budget | 1.108153 | -0.000588 | -0.170660 | 0.001943 | 2.349698 | 5.047016 | 0.988469 | 0.026653 | 0.028391 | 0.003120 |
| ml_external_risk_budget_enhancement_index_only | 0.994062 | -0.114678 | -0.165269 | 0.007334 | 2.369553 | 4.630265 | 0.937141 | 0.193058 | 0.137299 | 0.008995 |
| ml_external_risk_budget_enhancement_index_margin | 0.991916 | -0.116825 | -0.154860 | 0.017743 | 2.425625 | 4.931834 | 0.929797 | 0.213864 | 0.156108 | 0.010295 |
| ml_external_risk_budget_enhancement_index_shibor | 0.987834 | -0.120906 | -0.166442 | 0.006161 | 2.362108 | 4.571538 | 0.935328 | 0.198504 | 0.139554 | 0.009035 |
| ml_external_risk_budget_enhancement_all | 0.988328 | -0.120413 | -0.154928 | 0.017675 | 2.421969 | 4.913509 | 0.928141 | 0.218262 | 0.158363 | 0.010035 |
| ml_external_early_stress_risk_budget_index_only | 0.993707 | -0.115033 | -0.164356 | 0.008247 | 2.309306 | 4.680716 | 0.948198 | 0.153678 | 0.098945 | 0.007991 |
| ml_external_early_stress_risk_budget_all | 0.993241 | -0.115500 | -0.147973 | 0.024629 | 2.398926 | 5.196725 | 0.935494 | 0.190970 | 0.133478 | 0.009007 |
| ml_external_soft_confirm_risk_budget | 1.072530 | -0.036211 | -0.173347 | -0.000745 | 2.391314 | 4.767263 | 0.970988 | 0.097576 | 0.081399 | 0.006080 |
| ml_external_soft_confirm_risk_budget_index_only | 1.091861 | -0.016880 | -0.172603 | -0.000000 | 2.410592 | 4.878534 | 0.972750 | 0.087190 | 0.080670 | 0.006080 |
| ml_external_soft_confirm_early_stress | 1.080351 | -0.028389 | -0.171342 | 0.001261 | 2.331990 | 4.884917 | 0.985624 | 0.047056 | 0.034847 | 0.003401 |
| ml_external_soft_confirm_early_stress_index_only | 1.093790 | -0.014951 | -0.170660 | 0.001943 | 2.331591 | 4.981110 | 0.990789 | 0.028739 | 0.023189 | 0.002775 |
| ml_external_soft_lite_risk_budget | 1.100701 | -0.008040 | -0.172789 | -0.000186 | 2.412902 | 4.915297 | 0.977044 | 0.073831 | 0.071151 | 0.005280 |
| ml_external_soft_lite_risk_budget_index_only | 1.106104 | -0.002637 | -0.172603 | 0.000000 | 2.419165 | 4.945588 | 0.977094 | 0.071990 | 0.072089 | 0.005424 |
| ml_external_soft_lite_early_stress | 1.092895 | -0.015845 | -0.170831 | 0.001772 | 2.331234 | 4.972500 | 0.991632 | 0.027979 | 0.021864 | 0.002675 |
| ml_external_soft_lite_early_stress_index_only | 1.100753 | -0.007988 | -0.170660 | 0.001943 | 2.337897 | 5.013173 | 0.992254 | 0.023421 | 0.021192 | 0.002618 |

## Best External Variants
- Best by total return: `ml_external_soft_lite_risk_budget_index_only` with total return 1.1061.
- Best by max drawdown: `ml_external_early_stress_risk_budget_all` with max drawdown -0.1480.
- Best by Sharpe: `ml_external_risk_budget_enhancement_index_margin` with Sharpe 2.4256.

## Conclusion
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
