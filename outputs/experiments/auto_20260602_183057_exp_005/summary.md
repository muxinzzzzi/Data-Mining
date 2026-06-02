# 实验总结

## 配置
- model: RandomForest
- horizon: 5
- feature_top_k: 80
- target_label: big_up_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.6755876062840431
- avg_excess_return_vs_buy_hold: 0.000865174245916478
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.001154546209620643
- avg_avoided_downside: 0.0016938024736424982

## 测试期最终表现
- test_total_return: 1.1039311877993496
- test_excess_return_vs_buy_hold: -0.0048094955807718165
- test_max_drawdown: -0.17260282010543804
- test_sharpe: 2.3266732379070882

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.27173608592715803 | -0.4897815582681906 | 0.5691327216655068 | -0.9105889817411068 | -0.46104760078474205 | -1.0623231905654447 | 0.5128205128205128 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | -0.27173608592715803 | -0.46104760078474205 | True | 0.0 | 0.0 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.46646684570156394 | 1.0973018745551757 | 0.3940384983754315 | 2.1572579005575436 | -0.16271471376900515 | 6.74371634339682 | 0.576 | 0.996256 | 0.961 | 1.0 | 12 | 0.07800000000000007 | 7.800000000000007e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.0034636386288619285 | 0.005081407420927495 | 0.001539768792065566 | 0.002595522737749434 | 0.0031041738848041674 | 0.004321850781829657 | 0.7182510552782237 | 0.4638713229638145 | -0.16271471376900504 | False | 0.0037440000000000034 | 0.096 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。