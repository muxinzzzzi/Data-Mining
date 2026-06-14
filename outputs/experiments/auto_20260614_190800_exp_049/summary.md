# 实验总结

## 配置
- model: ExtraTrees
- horizon: 5
- feature_top_k: 50
- target_label: big_up_label
- strategy_family: ml_big_up_index_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.7867726042034618
- avg_excess_return_vs_buy_hold: -0.006503135663692501
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.07383558715132636
- avg_avoided_downside: 0.06868084478205233

## 测试期最终表现
- test_total_return: 1.0346393935050107
- test_excess_return_vs_buy_hold: -0.07410128987511078
- test_max_drawdown: -0.16789584799303026
- test_sharpe: 2.3185873595833875

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_index_enhancement | 0.21339079487128942 | 0.4548627099393969 | 0.17105738015740143 | 2.384003930866632 | -0.06692146014679057 | 6.796963320012248 | 0.5403225806451613 | 0.9583548387096774 | 0.916 | 1.0 | 80 | 0.8539999999999995 | 0.0008539999999999994 | 0 | 0 | 0 | nan | 0 | 0 | 0.034196290988384434 | 0.01660131132906762 | -0.018448979659316814 | -0.021507532120753403 | -0.037493087694740634 | 0.011845305102083191 | -3.165227689082222 | 0.23489832699204283 | -0.07129994372538018 | False | 0.041645161290322565 | 0.6451612903225806 | 2023H2 |
| ml_big_up_index_enhancement | -0.24762479655950553 | -0.45306558281591025 | 0.5213264209258928 | -0.9109345445195185 | -0.43109697041703965 | -1.0509597930544916 | 0.5128205128205128 | 0.9160000000000001 | 0.916 | 0.916 | 117 | 0.08399999999999996 | 8.399999999999997e-05 | 0 | 0 | 0 | nan | 0 | 0 | 0.11651621864171635 | 0.1367278120760969 | 0.02012759343438056 | 0.0241112893676525 | 0.04335173970481974 | 0.047806474256326226 | 0.9068173375927845 | -0.27173608592715803 | -0.46104760078474205 | False | 0.08399999999999996 | 1.0 | 2024H1 |
| ml_big_up_index_enhancement | 0.4417581587258379 | 1.0297995903900028 | 0.37548731021179077 | 2.153201231177427 | -0.15810972449107108 | 6.513195780365543 | 0.576 | 0.9453759999999998 | 0.916 | 1.0 | 121 | 0.9859999999999992 | 0.0009859999999999992 | 0 | 0 | 0 | nan | 0 | 0 | 0.07079425182387829 | 0.05271341094099248 | -0.019066840882885808 | -0.0221131642379766 | -0.038438751219897836 | 0.024411152315284245 | -1.57463894876567 | 0.4638713229638145 | -0.16271471376900504 | False | 0.054624 | 0.968 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。