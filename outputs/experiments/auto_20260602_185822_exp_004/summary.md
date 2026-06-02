# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.423164886085176
- avg_excess_return_vs_buy_hold: -0.002810532348824705
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.006827880399802405
- avg_avoided_downside: 0.004583995268225205

## 测试期最终表现
- test_total_return: 1.0952857580263449
- test_excess_return_vs_buy_hold: -0.01345492535377657
- test_max_drawdown: -0.17260282010543837
- test_sharpe: 2.317151802507999

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.27207481195759664 | -0.4902925502239992 | 0.5629424109479422 | -0.9287360029809206 | -0.46104760078474205 | -1.063431518544896 | 0.5128205128205128 | 0.9932905982905983 | 0.94 | 1.0 | 18 | 0.6620000000000006 | 0.0006620000000000006 | 0 | 0 | 0 | nan | 0 | 0 | 0.014680670948777341 | 0.013216753483818382 | -0.0021259174649589595 | -0.0003387260304386075 | -0.004578899155296243 | 0.013980124375419092 | -0.3275292145002092 | -0.27173608592715803 | -0.46104760078474205 | False | 0.006709401709401716 | 0.15384615384615385 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.455778451947779 | 1.0638397670777655 | 0.39455797049770247 | 2.11745795716442 | -0.16315267971513703 | 6.520516665342054 | 0.576 | 0.99724 | 0.94 | 1.0 | 10 | 0.4260000000000004 | 0.00042600000000000043 | 0 | 0 | 0 | nan | 0 | 0 | 0.005802970250629874 | 0.0005352323208572318 | -0.0056937379297726425 | -0.008092871016035508 | -0.011478575666421655 | 0.0036405085149004085 | -3.1530143713276466 | 0.4638713229638145 | -0.16271471376900504 | False | 0.0027600000000000025 | 0.08 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。