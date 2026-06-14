# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_down_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.5508054850547124
- avg_excess_return_vs_buy_hold: -0.0018417087990549492
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.014881731845020604
- avg_avoided_downside: 0.013501601692481421

## 测试期最终表现
- test_total_return: 1.0854488461247151
- test_excess_return_vs_buy_hold: -0.02329183725540629
- test_max_drawdown: -0.17009978105739954
- test_sharpe: 2.330146848911176

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.2328124355924901 | 0.5025782403804713 | 0.1790376690554979 | 2.4659035833604523 | -0.07042306462539338 | 7.136557363043953 | 0.5403225806451613 | 0.9957096774193548 | 0.972 | 1.0 | 19 | 0.28000000000000025 | 0.00028000000000000025 | 0 | 0 | 0 | nan | 0 | 0 | 0.004334798377800617 | 0.002742533836346144 | -0.0018722645414544735 | -0.0020858913995527306 | -0.003804924713278446 | 0.0032819827976315963 | -1.15933718970868 | 0.23489832699204283 | -0.07129994372538018 | False | 0.004290322580645165 | 0.1532258064516129 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.2696570852033858 | -0.4866392306420054 | 0.5625242664785968 | -0.9171529263369917 | -0.45943893517035406 | -1.059203287726509 | 0.5128205128205128 | 0.9845641025641022 | 0.939 | 1.0 | 48 | 0.5720000000000005 | 0.0005720000000000005 | 0 | 0 | 0 | nan | 0 | 0 | 0.019343992802385534 | 0.020995553826914495 | 0.001079561024528961 | 0.0020790007237722063 | 0.002325208360523929 | 0.011473241311951217 | 0.20266359760967045 | -0.27173608592715803 | -0.46104760078474205 | False | 0.015435897435897449 | 0.41025641025641024 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4583530872424302 | 1.0755382177666695 | 0.39050384127326265 | 2.144454073118814 | -0.15815966435303852 | 6.800331944090943 | 0.576 | 0.9812239999999999 | 0.939 | 1.0 | 72 | 0.5230000000000005 | 0.0005230000000000005 | 0 | 0 | 0 | nan | 0 | 0 | 0.020966404354875658 | 0.016766717414183622 | -0.004722686940692037 | -0.005518235721384324 | -0.009520936872435149 | 0.008192979375520961 | -1.1620848089623987 | 0.4638713229638145 | -0.16271471376900504 | False | 0.018776000000000015 | 0.576 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。