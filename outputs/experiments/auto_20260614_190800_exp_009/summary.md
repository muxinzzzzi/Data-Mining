# 实验总结

## 配置
- model: HistGradientBoosting
- horizon: 20
- feature_top_k: 50
- target_label: reduce_position_worth_label
- strategy_family: ml_big_up_index_enhancement
- search_engine: optuna

## 验证期最优依据
- robust_score: 0.7017470613777792
- avg_excess_return_vs_buy_hold: -0.006558026126131433
- positive_fold_ratio: 0.3333333333333333
- avg_missed_upside: 0.10846665830434703
- avg_avoided_downside: 0.10273010869218062

## 测试期最终表现
- test_total_return: 1.0060474757879945
- test_excess_return_vs_buy_hold: -0.10269320759212697
- test_max_drawdown: -0.1707164760605544
- test_sharpe: 2.3066455102619665

## 是否跑赢买入持有
- 否

## 验证折表现
| strategy | total_return | annualized_return | annualized_volatility | sharpe | max_drawdown | calmar | win_rate | average_position | minimum_position | maximum_position | days_below_full_exposure | total_turnover | total_transaction_cost | buy_signal_count | sell_signal_count | hold_signal_count | win_rate_after_buy | successful_sell_count | failed_sell_count | missed_upside | avoided_downside | net_timing_contribution | excess_return_vs_buy_hold | annualized_excess_return | tracking_error | information_ratio | benchmark_total_return | benchmark_max_drawdown | benchmark_clone | avg_abs_position_gap_from_1 | reduced_exposure_day_ratio | dataset_period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ml_big_up_index_enhancement | 0.20633472917924855 | 0.4377208702614912 | 0.17050375331068518 | 2.321559113386507 | -0.06766512850036799 | 6.468928382499277 | 0.5403225806451613 | 0.9484354838709678 | 0.875 | 1.0 | 72 | 0.7710000000000001 | 0.0007710000000000002 | 0 | 0 | 0 | nan | 0 | 0 | 0.04184702738887303 | 0.018280561152524855 | -0.02433746623634818 | -0.028563597812794272 | -0.049460012028707576 | 0.013669841833771929 | -3.6181846600824965 | 0.23489832699204283 | -0.07129994372538018 | False | 0.051564516129032276 | 0.5806451612903226 | 2023H2 |
| ml_big_up_index_enhancement | -0.23588181506834716 | -0.43469690824701956 | 0.49799240528266925 | -0.9111272848056875 | -0.41596783887062194 | -1.045025282308479 | 0.5128205128205128 | 0.875 | 0.875 | 0.875 | 117 | 0.125 | 0.000125 | 0 | 0 | 0 | nan | 0 | 0 | 0.17338723012160176 | 0.2034640060656205 | 0.029951775944018752 | 0.03585427085881088 | 0.06451151741788654 | 0.0711405866909617 | 0.906817337592784 | -0.27173608592715803 | -0.46104760078474205 | False | 0.125 | 1.0 | 2024H1 |
| ml_big_up_index_enhancement | 0.4369065715394036 | 1.0188907347080147 | 0.36547604623803975 | 2.1833437831234965 | -0.15590220516794662 | 6.535447870095284 | 0.576 | 0.9077199999999997 | 0.875 | 0.99 | 125 | 0.575 | 0.000575 | 0 | 0 | 0 | nan | 0 | 0 | 0.11016571740256628 | 0.08644575885839646 | -0.02429495854416982 | -0.026964751424410904 | -0.048978636425046365 | 0.037677455765627116 | -1.2999454296945718 | 0.4638713229638145 | -0.16271471376900504 | False | 0.09228000000000003 | 1.0 | 2024H2 |

## 诚实分析
- 当前实验没有跑赢买入持有，需要结合验证稳定性、上涨期机会成本和防守收益一起看。
- 若 avg_missed_upside 明显大于 avg_avoided_downside，说明仓位层仍然偏保守。
- 若验证折 robust_score 本身不高，则主要问题更可能在预测信号质量或跨阶段稳定性。