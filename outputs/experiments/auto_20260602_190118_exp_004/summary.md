# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 5
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_big_up_index_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.6893470051020195
- avg_excess_return_vs_buy_hold: -0.0089992442658835
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.09504860524172798
- avg_avoided_downside: 0.08794565848290392

## 测试期最终表现
- test_total_return: 0.9990756211802332
- test_excess_return_vs_buy_hold: -0.10966506219988825
- test_max_drawdown: -0.17009978105739987
- test_sharpe: 2.276516130635987

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_index_enhancement | 0.2104037622896866 | 0.4475934868812803 | 0.16885194036275503 | 2.3832608233996555 | -0.06602414925373223 | 6.779238989678896 | 0.5403225806451613 | 0.9508629032258066 | 0.893 | 1.0 | 76 | 0.7609999999999997 | 0.0007609999999999996 | 0 | 0 | 0 | nan | 0 | 0 | 0.04084464266894387 | 0.020507761095339132 | -0.021097881573604742 | -0.024494564702356225 | -0.04287633997216446 | 0.014362073973476103 | -2.98538637604489 | 0.23489832699204283 | -0.07129994372538018 | False | 0.04913709677419354 | 0.6129032258064516 | 2023H2 |
| ml_big_up_index_enhancement | -0.24103457347825152 | -0.44279628725686926 | 0.5082366061779322 | -0.9110404897224665 | -0.42265180129238933 | -1.0476621320502644 | 0.5128205128205128 | 0.8930000000000002 | 0.893 | 0.893 | 117 | 0.10699999999999998 | 0.00010699999999999999 | 0 | 0 | 0 | nan | 0 | 0 | 0.14841946898409106 | 0.17416518919217114 | 0.02563872020808008 | 0.03070151244890651 | 0.05522185890971091 | 0.060896342207463205 | 0.9068173375927847 | -0.27173608592715803 | -0.46104760078474205 | False | 0.10699999999999996 | 1.0 | 2024H1 |
| ml_big_up_index_enhancement | 0.4306666424196137 | 1.0000192861109913 | 0.36954447816348 | 2.1395979534165153 | -0.1571716187994342 | 6.362594555872776 | 0.576 | 0.9238720000000001 | 0.893 | 0.985 | 125 | 1.1899999999999995 | 0.0011899999999999992 | 0 | 0 | 0 | nan | 0 | 0 | 0.09588170407214905 | 0.06916402516120151 | -0.027907678910947534 | -0.03320468054420078 | -0.05626188068447028 | 0.031789682814024674 | -1.769815729638208 | 0.4638713229638145 | -0.16271471376900504 | False | 0.076128 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。