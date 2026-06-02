# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.426531237593735
- avg_excess_return_vs_buy_hold: -0.006984379545544021
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.018167002034023726
- avg_avoided_downside: 0.012608260340538765

## 测试期最终表现
- test_total_return: 1.0587657721515589
- test_excess_return_vs_buy_hold: -0.04997491122856257
- test_max_drawdown: -0.17260282010543804
- test_sharpe: 2.2950525107768054

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2711856978609565 | -0.4889506743133517 | 0.5588561207308572 | -0.9349103899852975 | -0.46104760078474205 | -1.0605210253368986 | 0.5128205128205128 | 0.9886495726495726 | 0.92 | 1.0 | 27 | 0.8639999999999994 | 0.0008639999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.023762707503280064 | 0.022660732229857577 | -0.001965975273422487 | 0.000550388066201557 | -0.004234408281217628 | 0.021247681777776535 | -0.19928801294673462 | -0.27173608592715803 | -0.46104760078474205 | False | 0.011350427350427347 | 0.23076923076923078 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4423677962609809 | 1.0268010419347546 | 0.3878774083853341 | 2.09899771589593 | -0.1611596444575869 | 6.371328538174967 | 0.576 | 0.9804159999999998 | 0.92 | 1.0 | 117 | 0.6879999999999995 | 0.0006879999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.030738298598791118 | 0.01516404879175872 | -0.0162622498070324 | -0.02150352670283362 | -0.032784695610977306 | 0.009347600633288104 | -3.5072845853326737 | 0.4638713229638145 | -0.16271471376900504 | False | 0.01958400000000001 | 0.936 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。