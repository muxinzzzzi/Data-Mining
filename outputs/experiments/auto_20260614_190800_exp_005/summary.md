# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3580857222551967
- avg_excess_return_vs_buy_hold: -0.0040387622898107045
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.01689064613330662
- avg_avoided_downside: 0.013765286279792871

## 测试期最终表现
- test_total_return: 1.084663633966314
- test_excess_return_vs_buy_hold: -0.024077049413807394
- test_max_drawdown: -0.17009978105739965
- test_sharpe: 2.3292997972337424

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2691992766882273 | -0.48594588237191694 | 0.5619617924790234 | -0.916327251032473 | -0.46104760078474205 | -1.0540037114276182 | 0.5128205128205128 | 0.9827692307692306 | 0.947 | 1.0 | 77 | 0.29800000000000026 | 0.00029800000000000025 | 0 | 0 | 0 | nan | 0 | 0 | 0.021219639540468533 | 0.0230521414376686 | 0.0015345018972000677 | 0.0025368092389307373 | 0.0033050810093539934 | 0.012743777417669931 | 0.25934861391814024 | -0.27173608592715803 | -0.46104760078474205 | False | 0.017230769230769247 | 0.6581196581196581 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44921822685545165 | 1.0488500875736029 | 0.38754224491319506 | 2.1251879309960047 | -0.1594817341876974 | 6.576615766788623 | 0.576 | 0.9789199999999998 | 0.947 | 0.982 | 125 | 0.3680000000000003 | 0.0003680000000000003 | 0 | 0 | 0 | nan | 0 | 0 | 0.029452298859451324 | 0.018243717401710012 | -0.011576581457741313 | -0.01465309610836285 | -0.023338388218806433 | 0.008286794118696202 | -2.8163349884790385 | 0.4638713229638145 | -0.16271471376900504 | False | 0.021080000000000012 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。