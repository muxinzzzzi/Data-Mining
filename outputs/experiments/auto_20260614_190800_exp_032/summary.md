# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.1045547799546214
- avg_excess_return_vs_buy_hold: -0.006149646019617723
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.02219637491337707
- avg_avoided_downside: 0.016678257114724693

## 测试期最终表现
- test_total_return: 1.085721118761311
- test_excess_return_vs_buy_hold: -0.023019564618810495
- test_max_drawdown: -0.1700997810573992
- test_sharpe: 2.32356546804113

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27222404550391366 | -0.4905175924438665 | 0.5620009945985842 | -0.9319453051145856 | -0.46104760078474205 | -1.0639196291423358 | 0.5128205128205128 | 0.9785213675213678 | 0.942 | 1.0 | 77 | 0.27500000000000024 | 0.0002750000000000003 | 0 | 0 | 0 | nan | 0 | 0 | 0.026159407781160993 | 0.023877027868439045 | -0.0025573799127219477 | -0.0004879595767556255 | -0.005508202888939633 | 0.011506865140444024 | -0.4786884022460251 | -0.27173608592715803 | -0.46104760078474205 | False | 0.0214786324786325 | 0.6581196581196581 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44591034448171696 | 1.0397826351768087 | 0.38413608304782376 | 2.128605781653232 | -0.15837641531520041 | 6.565261835908052 | 0.576 | 0.97052 | 0.942 | 0.973 | 125 | 0.24400000000000022 | 0.00024400000000000018 | 0 | 0 | 0 | nan | 0 | 0 | 0.04042971695897022 | 0.026157743475735042 | -0.014515973483235177 | -0.017960978482097545 | -0.029264202542202127 | 0.01139089539556003 | -2.569087110887595 | 0.4638713229638145 | -0.16271471376900504 | False | 0.029480000000000027 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。