# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 5
- feature_top_k: 50
- target_label: big_down_label
- strategy_family: ml_big_up_plus_115
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.6058947946240525
- avg_excess_return_vs_buy_hold: -0.0074396660097051
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.13404488377712165
- avg_avoided_downside: 0.1266307992495093

## 测试期最终表现
- test_total_return: 1.006807778586666
- test_excess_return_vs_buy_hold: -0.1019329047934554
- test_max_drawdown: -0.16722512755044128
- test_sharpe: 2.3085452773520134

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_plus_115 | 0.2029792696762034 | 0.4296054000473273 | 0.16800064062130654 | 2.3199515595859963 | -0.0661856244682767 | 6.490917076007051 | 0.5403225806451613 | 0.9473870967741935 | 0.847 | 1.15 | 75 | 2.6629999999999994 | 0.002662999999999999 | 0 | 0 | 0 | nan | 0 | 0 | 0.054800578496023704 | 0.02550481865557999 | -0.03195875984044371 | -0.03191905731583944 | -0.05554120617288686 | 0.018971102032646132 | -2.927674210876607 | 0.23489832699204283 | -0.07129994372538018 | False | 0.08406451612903225 | 0.6048387096774194 | 2023H2 |
| ml_big_up_plus_115 | -0.22787607245251584 | -0.4219910637472707 | 0.48205698535012936 | -0.9112696248108163 | -0.4054388176606707 | -1.04082550896854 | 0.5128205128205128 | 0.8469999999999994 | 0.847 | 0.847 | 117 | 0.15300000000000002 | 0.00015300000000000003 | 0 | 0 | 0 | nan | 0 | 0 | 0.21222596966884058 | 0.24903994342431957 | 0.036660973755478984 | 0.04386001347464219 | 0.07896209731949314 | 0.08707607810973711 | 0.9068173375927844 | -0.27173608592715803 | -0.46104760078474205 | False | 0.15299999999999997 | 1.0 | 2024H1 |
| ml_big_up_plus_115 | 0.42961136877589645 | 1.0002016519832622 | 0.35889084313408487 | 2.1880809566666124 | -0.15447203053202752 | 6.474969277858266 | 0.576 | 0.8872559999999995 | 0.847 | 0.985 | 125 | 0.8230000000000002 | 0.0008230000000000004 | 0 | 0 | 0 | nan | 0 | 0 | 0.13510810316650065 | 0.10534763566862838 | -0.03058346749787227 | -0.03425995418791805 | -0.061656270475710484 | 0.046130274952796514 | -1.3365684583237618 | 0.4638713229638145 | -0.16271471376900504 | False | 0.11274400000000007 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。