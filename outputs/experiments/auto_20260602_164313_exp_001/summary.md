# 实验总结

## 配置
- model: ExtraTrees
- horizon: 10
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_conservative_full_participation_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 1.3101845339488352
- avg_excess_return_vs_buy_hold: -0.008240205788206723
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.02644046101190109
- avg_avoided_downside: 0.020395150585276984

## 测试期最终表现
- test_total_return: 1.0915544453000523
- test_excess_return_vs_buy_hold: -0.01718623808006914
- test_max_drawdown: -0.16759407461569586
- test_sharpe: 2.3528130158316816

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_conservative_full_participation_enhancement | 0.23456098086814148 | 0.5069124871925459 | 0.17903098527045394 | 2.482075836631348 | -0.06957949575942457 | 7.285371669625594 | 0.5403225806451613 | 0.9927419354838709 | 0.97 | 1.0 | 30 | 0.3600000000000003 | 0.00036000000000000035 | 0 | 0 | 0 | nan | 0 | 0 | 0.00498262696485796 | 0.004886940818592134 | -0.00045568614626582553 | -0.00033734612390134977 | -0.0009260718456369935 | 0.0033817812849271513 | -0.2738414366903513 | 0.23489832699204283 | -0.07129994372538018 | False | 0.007258064516129039 | 0.24193548387096775 | 2023H2 |
| ml_conservative_full_participation_enhancement | -0.26954568738013274 | -0.4864705654536535 | 0.5614845793592061 | -0.9192704773035574 | -0.45960927375040117 | -1.0584437548095251 | 0.5128205128205128 | 0.9814529914529915 | 0.92 | 1.0 | 54 | 0.8499999999999999 | 0.0008499999999999998 | 0 | 0 | 0 | nan | 0 | 0 | 0.023939088559011404 | 0.025759347380460727 | 0.0009702588214493236 | 0.0021903985470252962 | 0.0020897882308139205 | 0.015033563140676092 | 0.1390081786505829 | -0.27173608592715803 | -0.46104760078474205 | False | 0.01854700854700856 | 0.46153846153846156 | 2024H1 |
| ml_conservative_full_participation_enhancement | 0.4372976531760704 | 1.0168827637021969 | 0.3820440796811904 | 2.106571248421461 | -0.15863112547280456 | 6.41036089652235 | 0.576 | 0.9641600000000002 | 0.92 | 1.0 | 116 | 1.0399999999999996 | 0.0010399999999999995 | 0 | 0 | 0 | nan | 0 | 0 | 0.05039966751183391 | 0.030539163556778087 | -0.020900503955055823 | -0.026573669787744114 | -0.04213541597339253 | 0.014717845881240641 | -2.8628792768579205 | 0.4638713229638145 | -0.16271471376900504 | False | 0.03584000000000001 | 0.928 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。