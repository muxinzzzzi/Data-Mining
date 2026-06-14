# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_up_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.6568527526352657
- avg_excess_return_vs_buy_hold: 0.0009232185294570187
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.013087196466184411
- avg_avoided_downside: 0.013795630460622384

## 测试期最终表现
- test_total_return: 1.1078221337995662
- test_excess_return_vs_buy_hold: -0.0009185495805552613
- test_max_drawdown: -0.16257466798180742
- test_sharpe: 2.371711989060327

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2670890431594427 | -0.48274346399810153 | 0.5587911781230114 | -0.9136178984339892 | -0.46104760078474205 | -1.0470577510357526 | 0.5128205128205128 | 0.9835897435897435 | 0.94 | 1.0 | 32 | 0.2400000000000002 | 0.00024000000000000022 | 0 | 0 | 0 | nan | 0 | 0 | 0.024454946409099787 | 0.028281258116157632 | 0.003586311707057845 | 0.004647042767715326 | 0.0077243636767399815 | 0.0190110338352827 | 0.406309501296257 | -0.27173608592715803 | -0.46104760078474205 | False | 0.016410256410256424 | 0.27350427350427353 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.46199393578447023 | 1.085926228152045 | 0.39219450935312417 | 2.1498168744050576 | -0.16236632616578717 | 6.6881246487232815 | 0.576 | 0.9846399999999998 | 0.94 | 1.0 | 32 | 0.18000000000000016 | 0.00018000000000000017 | 0 | 0 | 0 | nan | 0 | 0 | 0.014806642989453448 | 0.013105633265709517 | -0.0018810097237439312 | -0.0018773871793442698 | -0.0037921156030677663 | 0.008677021368068267 | -0.4370296490248232 | 0.4638713229638145 | -0.16271471376900504 | False | 0.015360000000000014 | 0.256 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。