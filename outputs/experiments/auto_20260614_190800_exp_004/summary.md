# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_bull_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.7048742668068624
- avg_excess_return_vs_buy_hold: 9.883543276349253e-05
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.0005645124801649947
- avg_avoided_downside: 0.0009303600575551497

## 测试期最终表现
- test_total_return: 1.1127012123031261
- test_excess_return_vs_buy_hold: 0.003960528923004691
- test_max_drawdown: -0.17260282010543804
- test_sharpe: 2.332921486490729

## 是否跑赢买入持有
- 是

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_bull_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_bull_full_participation_enhancement | -0.2705440074885186 | -0.48811872061994843 | 0.5689439991163543 | -0.9049002626084066 | -0.4601584546914562 | -1.0607622562259342 | 0.5128205128205128 | 0.9968034188034188 | 0.966 | 1.0 | 11 | 0.06800000000000006 | 6.800000000000007e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.00114060350366754 | 0.0027910801726654492 | 0.0015824766689979091 | 0.0011920784386394434 | 0.0034084112870724215 | 0.001988429070836482 | 1.7141226393549902 | -0.27173608592715803 | -0.46104760078474205 | False | 0.0031965811965811992 | 0.09401709401709402 | 2024H1 |
| ml_bull_full_participation_enhancement | 0.46297575082346554 | 1.0868908410365155 | 0.39521663232776777 | 2.1398054076363437 | -0.16271471376900504 | 6.679732987021076 | 0.576 | 0.9997280000000001 | 0.966 | 1.0 | 1 | 0.06800000000000006 | 6.800000000000007e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.000552933936827444 | 0.0 | -0.000620933936827444 | -0.0008955721403489658 | -0.001251802816644126 | 0.0008343718401717105 | -1.5002937016504654 | 0.4638713229638145 | -0.16271471376900504 | True | 0.0002720000000000002 | 0.008 | 2024H2 |
