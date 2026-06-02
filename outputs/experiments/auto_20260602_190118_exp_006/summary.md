# 实验总结

## 配置
- model: ExtraTrees
- horizon: 5
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.6395812282380058
- avg_excess_return_vs_buy_hold: -0.0021753966462163543
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.004193045887757721
- avg_avoided_downside: 0.0032009972618228296

## 测试期最终表现
- test_total_return: 1.0957651043110155
- test_excess_return_vs_buy_hold: -0.012975579069105958
- test_max_drawdown: -0.17260282010543804
- test_sharpe: 2.314038231365602

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.2699697272959458 | -0.48711243715931407 | 0.5677551444594331 | -0.9050365994529188 | -0.46104760078474205 | -1.0565339377760723 | 0.5128205128205128 | 0.9966923076923077 | 0.957 | 1.0 | 9 | 0.4300000000000004 | 0.0004300000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.003823551120450713 | 0.006299565514562941 | 0.0020460143941122277 | 0.001766358631212217 | 0.004406800233472489 | 0.005861221707694905 | 0.751856942672382 | -0.27173608592715803 | -0.46104760078474205 | False | 0.0033076923076923105 | 0.07692307692307693 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.4555787743939532 | 1.0632691165764778 | 0.39385360487601545 | 2.1198401345445834 | -0.16332220817084342 | 6.510254352330601 | 0.576 | 0.99484 | 0.957 | 1.0 | 15 | 0.5160000000000005 | 0.0005160000000000005 | 0 | 0 | 0 | nan | 0 | 0 | 0.00875558654282245 | 0.0033034262709055475 | -0.005968160271916902 | -0.00829254856986128 | -0.012031811108184486 | 0.005076245632368409 | -2.370218460561539 | 0.4638713229638145 | -0.16271471376900504 | False | 0.005160000000000005 | 0.12 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。