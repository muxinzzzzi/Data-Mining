# 实验总结

## 配置
- model: LightGBM
- horizon: 10
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_ultra_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.4506946844477673
- avg_excess_return_vs_buy_hold: -0.0015226686243714853
- positive_fold_ratio: 0.0
- avg_missed_upside: 0.005835362509469844
- avg_avoided_downside: 0.004616934987173824

## 测试期最终表现
- test_total_return: 1.079965416085638
- test_excess_return_vs_buy_hold: -0.02877526729448343
- test_max_drawdown: -0.17260282010543826
- test_sharpe: 2.296481549084578

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_ultra_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_ultra_conservative_full_participation_enhancement | -0.2718620066296934 | -0.48997155050429797 | 0.5644977151274976 | -0.9235742963264516 | -0.46104760078474205 | -1.0627352786790885 | 0.5128205128205128 | 0.9936153846153846 | 0.947 | 1.0 | 15 | 0.5060000000000004 | 0.0005060000000000005 | 0 | 0 | 0 | nan | 0 | 0 | 0.01225589661559142 | 0.011318156298170363 | -0.001443740317421058 | -0.0001259207025353648 | -0.003109594529829957 | 0.011638663997742923 | -0.26717796221568024 | -0.27173608592715803 | -0.46104760078474205 | False | 0.0063846153846153905 | 0.1282051282051282 | 2024H1 |
| ml_ultra_conservative_full_participation_enhancement | 0.4594292377932354 | 1.074287221933918 | 0.3944625177291567 | 2.130717558471782 | -0.1633787168189622 | 6.5754416661523996 | 0.576 | 0.995672 | 0.947 | 1.0 | 11 | 0.48200000000000043 | 0.00048200000000000055 | 0 | 0 | 0 | nan | 0 | 0 | 0.005250190912818112 | 0.002532648663351109 | -0.0031995422494670036 | -0.004442085170579091 | -0.006450277174925476 | 0.0041944980159726135 | -1.5377947850643567 | 0.4638713229638145 | -0.16271471376900504 | False | 0.004328000000000004 | 0.088 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。