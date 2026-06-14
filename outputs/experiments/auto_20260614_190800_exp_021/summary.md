# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_up_label
- strategy_family: ml_direct_signal_timing
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.4180599236709065
- avg_excess_return_vs_buy_hold: -0.019981910875191116
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.12400516392383576
- avg_avoided_downside: 0.10794966308872322

## 测试期最终表现
- test_total_return: 0.9488548928492455
- test_excess_return_vs_buy_hold: -0.15988579053087593
- test_max_drawdown: -0.13901030727345598
- test_sharpe: 2.37732829083164

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_direct_signal_timing | 0.22150498871535862 | 0.47470272637502386 | 0.16366539886522702 | 2.566935733005631 | -0.0637926195228995 | 7.441342429975318 | 0.5403225806451613 | 0.8693467741935482 | 0.8 | 1.0 | 113 | 1.6949999999999994 | 0.0016949999999999995 | 11 | 50 | 63 | 0.5454545454545454 | 21 | 29 | 0.0687693730823191 | 0.05807618566186111 | -0.012388187420457993 | -0.013393338276684208 | -0.025175993789963023 | 0.02263274309313484 | -1.1123704133592018 | 0.23489832699204283 | -0.07129994372538018 | False | 0.13065322580645158 | 0.9112903225806451 | 2023H2 |
| ml_direct_signal_timing | -0.24590499428033075 | -0.45039537109706385 | 0.5179116863845409 | -0.9109616665528538 | -0.4289041521865967 | -1.0501072764180592 | 0.5128205128205128 | 0.9099999999999999 | 0.91 | 0.91 | 117 | 0.08999999999999997 | 8.999999999999997e-05 | 0 | 117 | 0 | nan | 57 | 60 | 0.12483880568755322 | 0.1464940843672467 | 0.021565278679693482 | 0.025831091646827287 | 0.04644829254087827 | 0.0512212224174924 | 0.9068173375927837 | -0.27173608592715803 | -0.46104760078474205 | False | 0.08999999999999996 | 1.0 | 2024H1 |
| ml_direct_signal_timing | 0.3914878369680981 | 0.890035285329146 | 0.34124893649554644 | 2.1238916208007157 | -0.13491341135644552 | 6.597085318505841 | 0.576 | 0.874848 | 0.8 | 0.939 | 125 | 1.4679999999999995 | 0.0014679999999999995 | 0 | 84 | 41 | nan | 37 | 47 | 0.17840731300163493 | 0.11927871923706188 | -0.06059659376457305 | -0.07238348599571642 | -0.12216273302937929 | 0.05898393942964639 | -2.0711185826285807 | 0.4638713229638145 | -0.16271471376900504 | False | 0.12515199999999996 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。