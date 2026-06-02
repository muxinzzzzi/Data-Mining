# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 5
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.1738967645389806
- avg_excess_return_vs_buy_hold: -0.004280591899322357
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.027933871111780178
- avg_avoided_downside: 0.023295005331169006

## 测试期最终表现
- test_total_return: 1.0555111782470248
- test_excess_return_vs_buy_hold: -0.053229505133096655
- test_max_drawdown: -0.1672597790001077
- test_sharpe: 2.2931706380400962

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27269429580260596 | -0.4912263755035339 | 0.5565059009348953 | -0.9491902382269229 | -0.46104760078474205 | -1.0654569607724342 | 0.5128205128205128 | 0.9740170940170942 | 0.92 | 1.0 | 71 | 0.5119999999999993 | 0.0005119999999999993 | 0 | 0 | 0 | nan | 0 | 0 | 0.03838324867111241 | 0.03425982790611742 | -0.004635420764994993 | -0.000958209875447924 | -0.009983983186142997 | 0.02181107698819577 | -0.4577482896212032 | -0.27173608592715803 | -0.46104760078474205 | False | 0.02598290598290599 | 0.6068376068376068 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.45198775714129535 | 1.0530200678096806 | 0.38197030030723034 | 2.1607035866837974 | -0.15846840186582267 | 6.644984460064706 | 0.576 | 0.963264 | 0.92 | 1.0 | 121 | 0.927999999999999 | 0.0009279999999999991 | 0 | 0 | 0 | nan | 0 | 0 | 0.04541836466422811 | 0.035625188087389605 | -0.010721176576838506 | -0.011883565822519149 | -0.02161389197890644 | 0.014128131221334609 | -1.5298479070089421 | 0.4638713229638145 | -0.16271471376900504 | False | 0.03673600000000002 | 0.968 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。