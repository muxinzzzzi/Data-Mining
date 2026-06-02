# 实验总结

## 配置
- model: LightGBM
- horizon: 10
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.6497034771689136
- avg_excess_return_vs_buy_hold: 0.00015322262301042944
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.00514223788777739
- avg_avoided_downside: 0.005834830108462659

## 测试期最终表现
- test_total_return: 1.083629950873091
- test_excess_return_vs_buy_hold: -0.02511073250703033
- test_max_drawdown: -0.17260282010543793
- test_sharpe: 2.3018844175800424

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.26790638285992163 | -0.4839850950196054 | 0.5635870805189502 | -0.905350252218231 | -0.46104760078474205 | -1.0497508157418491 | 0.5128205128205128 | 0.9912820512820512 | 0.94 | 1.0 | 20 | 0.4320000000000004 | 0.0004320000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.012990582018635911 | 0.017137926305357938 | 0.0037153442867220265 | 0.0038297030672364007 | 0.008002280002170514 | 0.013604359389224151 | 0.5882143931384982 | -0.27173608592715803 | -0.46104760078474205 | False | 0.008717948717948726 | 0.17094017094017094 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.4605012877656094 | 1.0773601573952258 | 0.39501512266233785 | 2.1320339025298694 | -0.16271471376900504 | 6.621160019521532 | 0.576 | 0.9984639999999999 | 0.952 | 1.0 | 4 | 0.28800000000000026 | 0.0002880000000000003 | 0 | 0 | 0 | nan | 0 | 0 | 0.0024361316446962593 | 0.0003665640200300405 | -0.0023575676246662194 | -0.0033700351982051124 | -0.00475285633132709 | 0.0023521451879009846 | -2.020647516052553 | 0.4638713229638145 | -0.16271471376900504 | False | 0.0015360000000000013 | 0.032 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。