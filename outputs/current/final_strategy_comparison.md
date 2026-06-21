# Final Strategy Comparison

Test period: 2025-01-01 to 2026-05-06

This table compares buy-and-hold, the original risk-budget/early-stress strategies, and the external-market ablation variants. All external variants are no-leverage and long-only.

| strategy | total_return | excess_return_vs_buy_hold | max_drawdown | drawdown_improvement_vs_buy_hold | sharpe | calmar | average_position | total_turnover | missed_upside | avoided_downside | total_transaction_cost | outperforms_buy_hold | reduces_max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| buy_hold | 1.108741 | 0.000000 | -0.172603 | 0.000000 | 2.325563 | 5.042679 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | False | False |
| ml_risk_budget_enhancement | 1.113495 | 0.004754 | -0.172603 | 0.000000 | 2.423439 | 4.978558 | 0.976406 | 5.000000 | 0.065942 | 0.069375 | 0.005000 | True | False |
| ml_early_stress_risk_budget | 1.108153 | -0.000588 | -0.170660 | 0.001943 | 2.349698 | 5.047016 | 0.988469 | 3.120000 | 0.026653 | 0.028391 | 0.003120 | False | True |
| ml_external_risk_budget_enhancement_index_only | 0.994062 | -0.114678 | -0.165269 | 0.007334 | 2.369553 | 4.630265 | 0.937141 | 8.995000 | 0.193058 | 0.137299 | 0.008995 | False | True |
| ml_external_risk_budget_enhancement_index_margin | 0.991916 | -0.116825 | -0.154860 | 0.017743 | 2.425625 | 4.931834 | 0.929797 | 10.295000 | 0.213864 | 0.156108 | 0.010295 | False | True |
| ml_external_risk_budget_enhancement_index_shibor | 0.987834 | -0.120906 | -0.166442 | 0.006161 | 2.362108 | 4.571538 | 0.935328 | 9.035000 | 0.198504 | 0.139554 | 0.009035 | False | True |
| ml_external_risk_budget_enhancement_all | 0.988328 | -0.120413 | -0.154928 | 0.017675 | 2.421969 | 4.913509 | 0.928141 | 10.035000 | 0.218262 | 0.158363 | 0.010035 | False | True |
| ml_external_early_stress_risk_budget_index_only | 0.993707 | -0.115033 | -0.164356 | 0.008247 | 2.309306 | 4.680716 | 0.948198 | 7.991000 | 0.153678 | 0.098945 | 0.007991 | False | True |
| ml_external_early_stress_risk_budget_all | 0.993241 | -0.115500 | -0.147973 | 0.024629 | 2.398926 | 5.196725 | 0.935494 | 9.007000 | 0.190970 | 0.133478 | 0.009007 | False | True |
| ml_external_soft_confirm_risk_budget | 1.072530 | -0.036211 | -0.173347 | -0.000745 | 2.391314 | 4.767263 | 0.970988 | 6.080000 | 0.097576 | 0.081399 | 0.006080 | False | False |
| ml_external_soft_confirm_risk_budget_index_only | 1.091861 | -0.016880 | -0.172603 | -0.000000 | 2.410592 | 4.878534 | 0.972750 | 6.080000 | 0.087190 | 0.080670 | 0.006080 | False | False |
| ml_external_soft_confirm_early_stress | 1.080351 | -0.028389 | -0.171342 | 0.001261 | 2.331990 | 4.884917 | 0.985624 | 3.400800 | 0.047056 | 0.034847 | 0.003401 | False | True |
| ml_external_soft_confirm_early_stress_index_only | 1.093790 | -0.014951 | -0.170660 | 0.001943 | 2.331591 | 4.981110 | 0.990789 | 2.775200 | 0.028739 | 0.023189 | 0.002775 | False | True |
| ml_external_soft_lite_risk_budget | 1.100701 | -0.008040 | -0.172789 | -0.000186 | 2.412902 | 4.915297 | 0.977044 | 5.280000 | 0.073831 | 0.071151 | 0.005280 | False | False |
| ml_external_soft_lite_risk_budget_index_only | 1.106104 | -0.002637 | -0.172603 | 0.000000 | 2.419165 | 4.945588 | 0.977094 | 5.424000 | 0.071990 | 0.072089 | 0.005424 | False | False |
| ml_external_soft_lite_early_stress | 1.092895 | -0.015845 | -0.170831 | 0.001772 | 2.331234 | 4.972500 | 0.991632 | 2.675200 | 0.027979 | 0.021864 | 0.002675 | False | True |
| ml_external_soft_lite_early_stress_index_only | 1.100753 | -0.007988 | -0.170660 | 0.001943 | 2.337897 | 5.013173 | 0.992254 | 2.617600 | 0.023421 | 0.021192 | 0.002618 | False | True |
