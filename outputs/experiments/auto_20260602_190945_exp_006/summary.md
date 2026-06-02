# 实验总结

## 配置
- model: RandomForest
- horizon: 10
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.1087266159000182
- avg_excess_return_vs_buy_hold: -0.007370528978128806
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.02135386926256445
- avg_avoided_downside: 0.015255965057612038

## 测试期最终表现
- test_total_return: 1.0608088567901262
- test_excess_return_vs_buy_hold: -0.047931826589995286
- test_max_drawdown: -0.17009978105739965
- test_sharpe: 2.305484672119901

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2732497094496247 | -0.49206283987600397 | 0.5619635808409177 | -0.9374082915997783 | -0.46104760078474205 | -1.0672712297785985 | 0.5128205128205128 | 0.9814273504273504 | 0.92 | 1.0 | 63 | 0.8499999999999993 | 0.0008499999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.025649881853240897 | 0.022533333571608055 | -0.0039665482816328415 | -0.001513623522466645 | -0.00854333476043996 | 0.014875990978027361 | -0.5743035723172275 | -0.27173608592715803 | -0.46104760078474205 | False | 0.018572649572649577 | 0.5384615384615384 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44327335955189473 | 1.0295757558204546 | 0.385807294755035 | 2.1114886567421873 | -0.1603004813557526 | 6.422786426545605 | 0.576 | 0.9723679999999999 | 0.92 | 1.0 | 124 | 0.8499999999999993 | 0.0008499999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.03841172593445245 | 0.02323456160122806 | -0.016027164333224386 | -0.020597963411919773 | -0.03231076329578028 | 0.011940969656720951 | -2.705874332206701 | 0.4638713229638145 | -0.16271471376900504 | False | 0.02763200000000001 | 0.992 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。