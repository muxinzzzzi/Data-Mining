# 实验总结

## 配置
- model: ExtraTrees
- horizon: 5
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_big_up_index_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.810291890562445
- avg_excess_return_vs_buy_hold: -0.006761734095726039
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.07495431144747418
- avg_avoided_downside: 0.06974039713732434

## 测试期最终表现
- test_total_return: 1.0310887481438282
- test_excess_return_vs_buy_hold: -0.07765193523629321
- test_max_drawdown: -0.17093442388758728
- test_sharpe: 2.2864105029419397

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_index_enhancement | 0.2174016903176086 | 0.4646526909325104 | 0.17391123938558645 | 2.3862711749124124 | -0.06823008400081965 | 6.810085283303015 | 0.5403225806451613 | 0.971548387096774 | 0.944 | 1.0 | 63 | 0.5040000000000004 | 0.0005040000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.025059737626925996 | 0.010656587111162399 | -0.014907150515763598 | -0.01749663667443424 | -0.030295176854616336 | 0.008569127961811028 | -3.535386213116329 | 0.23489832699204283 | -0.07129994372538018 | False | 0.028451612903225832 | 0.5080645161290323 | 2023H2 |
| ml_big_up_index_enhancement | -0.24619159919067812 | -0.45084083683414045 | 0.5184808087956412 | -0.9109571214265824 | -0.42927012391067687 | -1.0502497418803645 | 0.5128205128205128 | 0.9109999999999998 | 0.911 | 0.911 | 117 | 0.08899999999999997 | 8.899999999999997e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.1234517078465804 | 0.14486637231872176 | 0.02132566447214137 | 0.02554448673647991 | 0.045932200401535235 | 0.050652097723964694 | 0.9068173375927852 | -0.27173608592715803 | -0.46104760078474205 | False | 0.08899999999999998 | 1.0 | 2024H1 |
| ml_big_up_index_enhancement | 0.4355382706145907 | 1.0125297267359756 | 0.37469686665613716 | 2.133621183198352 | -0.1596342838017828 | 6.342808716411002 | 0.576 | 0.9421679999999998 | 0.911 | 1.0 | 122 | 0.8969999999999994 | 0.0008969999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.07635148886891613 | 0.053698231982088875 | -0.023550256886827253 | -0.028333052349223786 | -0.047477317883843746 | 0.02580416749232088 | -1.839908917734805 | 0.4638713229638145 | -0.16271471376900504 | False | 0.05783200000000001 | 0.976 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。