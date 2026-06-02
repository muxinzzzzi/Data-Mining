# 实验总结

## 配置
- model: RandomForest
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3660614763956138
- avg_excess_return_vs_buy_hold: -0.004366781083420553
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.017686476179024673
- avg_avoided_downside: 0.015103818735396584

## 测试期最终表现
- test_total_return: 1.0643209431937382
- test_excess_return_vs_buy_hold: -0.04441974018638328
- test_max_drawdown: -0.17060060232575003
- test_sharpe: 2.303938414988219

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23489832699204283 | 0.5077494213848086 | 0.18112757201442167 | 2.4584581432929404 | -0.07129994372538018 | 7.1213158784593595 | 0.5403225806451613 | 1.0 | 1.0 | 1.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0 | nan | 0 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | nan | 0.23489832699204283 | -0.07129994372538018 | True | 0.0 | 0.0 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.26702315221901785 | -0.4826432986088791 | 0.5602465791493241 | -0.9095205236623108 | -0.46104760078474205 | -1.0468404949670693 | 0.5128205128205128 | 0.983076923076923 | 0.92 | 1.0 | 63 | 0.9199999999999994 | 0.0009199999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.022644942539764772 | 0.02759968914109881 | 0.004034746601334038 | 0.004712933708140188 | 0.008690223449027201 | 0.01891533654504582 | 0.45942737673907724 | -0.27173608592715803 | -0.46104760078474205 | False | 0.016923076923076926 | 0.5384615384615384 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.44605804600541266 | 1.0369894916966773 | 0.38827905579003963 | 2.110531269384775 | -0.16143630080495164 | 6.4235211444145675 | 0.576 | 0.9783039999999997 | 0.92 | 1.0 | 124 | 0.9199999999999994 | 0.0009199999999999993 | 0 | 0 | 0 | nan | 0 | 0 | 0.030414485997309247 | 0.017711767065090943 | -0.013622718932218303 | -0.017813276958401847 | -0.0274634013673521 | 0.010635131640888555 | -2.5823282959436464 | 0.4638713229638145 | -0.16271471376900504 | False | 0.021696000000000003 | 0.992 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。