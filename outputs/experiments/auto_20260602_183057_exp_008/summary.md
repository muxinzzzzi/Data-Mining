# 实验总结

## 配置
- model: LightGBM
- horizon: 5
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_big_up_index_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.8160449335813321
- avg_excess_return_vs_buy_hold: -0.00958832909438164
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.07977610667483352
- avg_avoided_downside: 0.07185558986564844

## 测试期最终表现
- test_total_return: 1.018720579704901
- test_excess_return_vs_buy_hold: -0.0900201036752204
- test_max_drawdown: -0.1657548625006372
- test_sharpe: 2.304988210561182

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_index_enhancement | 0.20530701473519986 | 0.43598306408362 | 0.167831502761768 | 2.3455815470031456 | -0.06515573175371148 | 6.6914000096205655 | 0.5403225806451613 | 0.9418629032258067 | 0.9 | 1.0 | 107 | 0.8699999999999998 | 0.0008699999999999998 | 0 | 0 | 0 | nan | 0 | 0 | 0.046662097137629595 | 0.02212583628354052 | -0.025406260854089074 | -0.029591312256842972 | -0.05163207850992295 | 0.015222373834816139 | -3.391854586558088 | 0.23489832699204283 | -0.07129994372538018 | False | 0.058137096774193535 | 0.8629032258064516 | 2023H2 |
| ml_big_up_index_enhancement | -0.2479114700011299 | -0.4535100144248856 | 0.5218955433667789 | -0.910930058656109 | -0.431461737938743 | -1.0511013481554021 | 0.5128205128205128 | 0.917 | 0.917 | 0.917 | 117 | 0.08299999999999996 | 8.299999999999997e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.11512912080074351 | 0.13510010002757197 | 0.01988797922682846 | 0.023824615926028136 | 0.04283564756547666 | 0.04723734956279854 | 0.9068173375927846 | -0.27173608592715803 | -0.46104760078474205 | False | 0.08299999999999998 | 1.0 | 2024H1 |
| ml_big_up_index_enhancement | 0.4408730320114844 | 1.0272186378461279 | 0.3728652403237258 | 2.1624743439693743 | -0.15567674079557925 | 6.598407909855843 | 0.576 | 0.9404800000000003 | 0.9 | 1.0 | 122 | 0.9559999999999994 | 0.0009559999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.07753710208612744 | 0.05834083328583283 | -0.020152268800294602 | -0.02299829095233008 | -0.04062697390139391 | 0.025463867845400927 | -1.5954753672165172 | 0.4638713229638145 | -0.16271471376900504 | False | 0.05951999999999998 | 0.976 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。