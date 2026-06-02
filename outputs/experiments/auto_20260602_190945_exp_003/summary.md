# 实验总结

## 配置
- model: ExtraTrees
- horizon: 5
- feature_top_k: 50
- target_label: avoid_loss_label
- strategy_family: ml_direct_signal_timing
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.05825958773749734
- avg_excess_return_vs_buy_hold: -0.030633192605304133
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.18199791744878188
- avg_avoided_downside: 0.15680673688994237

## 测试期最终表现
- test_total_return: 0.9034280790722249
- test_excess_return_vs_buy_hold: -0.20531260430789655
- test_max_drawdown: -0.13676523950341912
- test_sharpe: 2.363903107776873

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_direct_signal_timing | 0.18360327003073862 | 0.3887759441198204 | 0.15457468725803586 | 2.2939823119109146 | -0.06182012980976481 | 6.288824454367471 | 0.5403225806451613 | 0.8442177419354839 | 0.8 | 1.0 | 111 | 3.014999999999999 | 0.0030149999999999986 | 13 | 49 | 62 | 0.6153846153846154 | 21 | 28 | 0.09911476587235703 | 0.057498152624836804 | -0.04463161324752022 | -0.05129505696130421 | -0.09070295595463786 | 0.029960081480236656 | -3.027460256223622 | 0.23489832699204283 | -0.07129994372538018 | False | 0.1557822580645161 | 0.8951612903225806 | 2023H2 |
| ml_direct_signal_timing | -0.2267334466241726 | -0.4201655018232511 | 0.4797804971759865 | -0.9112907302267814 | -0.403921479210534 | -1.0402157930409297 | 0.5128205128205128 | 0.8430000000000001 | 0.843 | 0.843 | 117 | 0.15700000000000003 | 0.00015700000000000002 | 0 | 117 | 0 | nan | 57 | 60 | 0.21777436103273187 | 0.25555079161841937 | 0.0376194305856875 | 0.045002639302985425 | 0.08102646587686546 | 0.0893525768838479 | 0.9068173375927837 | -0.27173608592715803 | -0.46104760078474205 | False | 0.15700000000000003 | 1.0 | 2024H1 |
| ml_direct_signal_timing | 0.3782641628062209 | 0.8582673699327665 | 0.32654008700164466 | 2.1453164311868735 | -0.13201503037764561 | 6.501285251214083 | 0.576 | 0.8347520000000003 | 0.8 | 0.94 | 125 | 0.8889999999999993 | 0.0008889999999999994 | 0 | 92 | 33 | nan | 43 | 49 | 0.2291046254412567 | 0.1573712664265709 | -0.0726223590146858 | -0.08560716015759362 | -0.1464066757736067 | 0.06951137026030503 | -2.106226293991118 | 0.4638713229638145 | -0.16271471376900504 | False | 0.165248 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。