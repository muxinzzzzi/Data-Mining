# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 5
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.6347295352836577
- avg_excess_return_vs_buy_hold: -0.0007082404617767546
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.007218203297905045
- avg_avoided_downside: 0.005741672314195724

## 测试期最终表现
- test_total_return: 1.0737083431753436
- test_excess_return_vs_buy_hold: -0.035032340204777856
- test_max_drawdown: -0.17260282010543826
- test_sharpe: 2.2919271576544635

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.27575400387550586 | -0.49582520336402147 | 0.5636110408902724 | -0.9461508926269793 | -0.46104760078474205 | -1.0754316962502029 | 0.5128205128205128 | 0.9923247863247864 | 0.94 | 1.0 | 17 | 0.5520000000000005 | 0.0005520000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.016599399829358614 | 0.010180101573445144 | -0.00697129825591347 | -0.004017917948347827 | -0.015015103935813675 | 0.012644176588722203 | -1.1875114073625157 | -0.27173608592715803 | -0.46104760078474205 | False | 0.007675213675213682 | 0.1452991452991453 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.46576451952683207 | 1.0924799983020894 | 0.3941337307515337 | 2.1543731924055063 | -0.16276056491991076 | 6.712190995648508 | 0.576 | 0.993296 | 0.94 | 1.0 | 17 | 0.9120000000000008 | 0.0009120000000000009 | 0 | 0 | 0 | nan | 0 | 0 | 0.005055210064356523 | 0.007044915369142025 | 0.001077705304785501 | 0.001893196563017563 | 0.002172653894447579 | 0.004680982174172272 | 0.46414487678149824 | 0.4638713229638145 | -0.16271471376900504 | False | 0.006704000000000006 | 0.136 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。