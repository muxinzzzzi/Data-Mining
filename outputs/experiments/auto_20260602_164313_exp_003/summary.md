# 实验总结

## 配置
- model: ExtraTrees
- horizon: 5
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.372447065659768
- avg_excess_return_vs_buy_hold: -0.005736503881742368
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.015093546742738759
- avg_avoided_downside: 0.011442021314541041

## 测试期最终表现
- test_total_return: 1.080677914471314
- test_excess_return_vs_buy_hold: -0.028062768908807545
- test_max_drawdown: -0.17060060232574992
- test_sharpe: 2.3165828741287675

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.26881379788091353 | -0.4853616882216142 | 0.5641550385789298 | -0.9084419493211445 | -0.46104760078474205 | -1.052736609832667 | 0.5128205128205128 | 0.987282051282051 | 0.92 | 1.0 | 73 | 0.7159999999999995 | 0.0007159999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.015501854176666616 | 0.018884656780321984 | 0.002666802603655368 | 0.0029222880462445033 | 0.005743882530950079 | 0.011669200062936447 | 0.49222590237301006 | -0.27173608592715803 | -0.46104760078474205 | False | 0.012717948717948721 | 0.6239316239316239 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4437395232723429 | 1.0304106251604437 | 0.3884414616126725 | 2.1014556649842606 | -0.1621267645112381 | 6.355586187553993 | 0.576 | 0.9805119999999998 | 0.92 | 1.0 | 118 | 0.8639999999999994 | 0.0008639999999999993 | 0 | 0 | 0 | nan | 0 | 0 | 0.02977878605154966 | 0.015441407163301136 | -0.015201378888248521 | -0.020131799691471608 | -0.03064597983870904 | 0.010372132757874474 | -2.9546459300227097 | 0.4638713229638145 | -0.16271471376900504 | False | 0.019488000000000005 | 0.944 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。