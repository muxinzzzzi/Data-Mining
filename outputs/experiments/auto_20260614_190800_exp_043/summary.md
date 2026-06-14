# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.6681405623340737
- avg_excess_return_vs_buy_hold: 0.001264166142053121
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.011996596760669045
- avg_avoided_downside: 0.012923883017568088

## 测试期最终表现
- test_total_return: 1.0839476513718402
- test_excess_return_vs_buy_hold: -0.0247930320082812
- test_max_drawdown: -0.17009978105739976
- test_sharpe: 2.3215298249305296

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2674724571713044 | -0.4833261121065813 | 0.5596356062933563 | -0.9133895574995128 | -0.46104760078474205 | -1.0483214993070549 | 0.5128205128205128 | 0.984957264957265 | 0.945 | 1.0 | 32 | 0.2200000000000002 | 0.0002200000000000002 | 0 | 0 | 0 | nan | 0 | 0 | 0.022417034208341475 | 0.025924486606477833 | 0.0032874523981363583 | 0.004263628755853621 | 0.007080666703678319 | 0.017426781015675803 | 0.40630950129625726 | -0.27173608592715803 | -0.46104760078474205 | False | 0.015042735042735057 | 0.27350427350427353 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.46340019263412024 | 1.089614938863817 | 0.3923551104447395 | 2.1540515482225198 | -0.1616805268985475 | 6.739308435996975 | 0.576 | 0.98556 | 0.945 | 1.0 | 33 | 0.16000000000000014 | 0.00016000000000000015 | 0 | 0 | 0 | nan | 0 | 0 | 0.013572756073665662 | 0.01284716244622643 | -0.0008855936274392328 | -0.00047113032969425817 | -0.0017853567529174945 | 0.008043749337124786 | -0.2219557917695711 | 0.4638713229638145 | -0.16271471376900504 | False | 0.014440000000000012 | 0.264 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。