# 实验总结

## 配置
- model: RandomForest
- horizon: 20
- feature_top_k: 80
- target_label: big_up_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.5400949626130693
- avg_excess_return_vs_buy_hold: -0.0024563657437504474
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.013453087272908568
- avg_avoided_downside: 0.011482007597014001

## 测试期最终表现
- test_total_return: 1.0839722226906296
- test_excess_return_vs_buy_hold: -0.02476846068949179
- test_max_drawdown: -0.17009978105739965
- test_sharpe: 2.3277712414191365

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.233543914805191 | 0.504390639791666 | 0.17943439174416526 | 2.4675592221941054 | -0.07064231291853607 | 7.140064062926758 | 0.5403225806451613 | 0.9964435483870966 | 0.979 | 1.0 | 21 | 0.2100000000000002 | 0.00021000000000000017 | 0 | 0 | 0 | nan | 0 | 0 | 0.003496228210830327 | 0.0024615210002669865 | -0.0012447072105633405 | -0.0013544121868518388 | -0.002529566266628717 | 0.002554296190713327 | -0.9903183020925606 | 0.23489832699204283 | -0.07129994372538018 | False | 0.003556451612903229 | 0.1693548387096774 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.27129684131776066 | -0.48911851849908694 | 0.5636503004763349 | -0.9227321886662454 | -0.4598413139640749 | -1.0636680603633177 | 0.5128205128205128 | 0.9855213675213674 | 0.93 | 1.0 | 48 | 0.6229999999999997 | 0.0006229999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.018051983242993182 | 0.017814991501860968 | -0.0008599917411322131 | 0.0004392446093973712 | -0.0018522899039770627 | 0.010420077072738482 | -0.17776163180434765 | -0.27173608592715803 | -0.46104760078474205 | False | 0.014478632478632474 | 0.41025641025641024 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.45741739331001763 | 1.0734940845649819 | 0.39126543680220865 | 2.137740475883283 | -0.15848497296546948 | 6.77347551934071 | 0.576 | 0.9835759999999999 | 0.93 | 1.0 | 73 | 0.5739999999999997 | 0.0005739999999999997 | 0 | 0 | 0 | nan | 0 | 0 | 0.018811050364902198 | 0.014169510288914052 | -0.005215540075988146 | -0.006453929653796875 | -0.010514528793192129 | 0.007561528971807545 | -1.3905294593718505 | 0.4638713229638145 | -0.16271471376900504 | False | 0.016424000000000005 | 0.584 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。