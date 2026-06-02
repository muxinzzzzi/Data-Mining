# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 5
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.1863358334588276
- avg_excess_return_vs_buy_hold: -0.0034850372331285406
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.02023795533678002
- avg_avoided_downside: 0.0162822208972524

## 测试期最终表现
- test_total_return: 1.0551891761637724
- test_excess_return_vs_buy_hold: -0.05355150721634905
- test_max_drawdown: -0.17260282010543815
- test_sharpe: 2.27278114144539

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27477499951536233 | -0.49435616575602415 | 0.558239987851762 | -0.9555026975328184 | -0.46104760078474205 | -1.0722453926982551 | 0.5128205128205128 | 0.9805982905982903 | 0.92 | 1.0 | 71 | 0.6379999999999995 | 0.0006379999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.03237257909870298 | 0.02597487288586722 | -0.007035706212835762 | -0.003038913588204295 | -0.015153828766107798 | 0.020615224964556512 | -0.7350794760746767 | -0.27173608592715803 | -0.46104760078474205 | False | 0.019401709401709398 | 0.6068376068376068 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4564551248526332 | 1.0657741949517696 | 0.3873276216679915 | 2.1521458141933674 | -0.16082356685234844 | 6.6269777235462834 | 0.576 | 0.9760639999999997 | 0.92 | 1.0 | 118 | 1.153999999999999 | 0.0011539999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.02834128691163708 | 0.02287178980588998 | -0.0066234971057470985 | -0.007416198111181327 | -0.013352970165186098 | 0.00976458825184317 | -1.367489321699313 | 0.4638713229638145 | -0.16271471376900504 | False | 0.023936000000000006 | 0.944 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。